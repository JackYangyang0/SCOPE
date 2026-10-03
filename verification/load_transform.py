"""Conservative transformations using scoped integer-expression evaluation.

This is deliberately not a general CUDA parser. Only const-int expressions and
recognized one-dimensional launches are supported; ambiguous code is rejected.
"""
import ast
import re

from SCOPE.verification.gemm_semantic_checker import extract_launch_config


def mask_comments(source):
    return re.sub(r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: re.sub(r'[^\n]', ' ', m[0]), source)


LOAD_LOOP = re.compile(
    r'for\s*\(int\s+(\w+)\s*=\s*([^;]+?)\s*;\s*\1\s*<\s*([^;]+?)\s*;'
    r'\s*\1\s*\+=\s*([^)]+?)\s*\)\s*\{'
)


def loop_end(code, match):
    depth, end = 1, match.end()
    while end < len(code) and depth:
        depth += (code[end] == '{') - (code[end] == '}')
        end += 1
    if depth:
        raise ValueError('Unbalanced producer loop')
    return end


class Expressions:
    def __init__(self, source):
        self.code = mask_comments(source)
        self.tile = extract_launch_config(source)
        self.scopes, stack = [], []
        for i, char in enumerate(self.code):
            self.scopes.append(tuple(stack))
            if char == '{':
                stack.append(i)
            elif char == '}' and stack:
                stack.pop()
        self.declarations = list(re.finditer(r'\bconst\s+int\s+(\w+)\s*=\s*([^;]+);', self.code))

    def value(self, expression, position, bindings=None, active=()):
        bindings = bindings or {}
        scope = self.scopes[position]

        def symbol(name):
            if name in bindings:
                return bindings[name]
            if name in self.tile:
                return self.tile[name]
            if name in active:
                raise ValueError('Cyclic integer expression')
            visible = [d for d in self.declarations if d[1] == name and d.start() < position
                       and scope[:len(self.scopes[d.start()])] == self.scopes[d.start()]]
            if not visible:
                raise ValueError('Unresolved integer expression: ' + name)
            declaration = visible[-1]
            return self.value(declaration[2], declaration.start(), bindings, (*active, name))

        def walk(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return node.value
            if isinstance(node, ast.Name):
                return symbol(node.id)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                return symbol(node.value.id + '.' + node.attr)
            if isinstance(node, ast.BinOp):
                a, b = walk(node.left), walk(node.right)
                if a < 0 or b < 0:
                    raise ValueError('Negative index arithmetic is unsupported')
                if isinstance(node.op, ast.Add): return a + b
                if isinstance(node.op, ast.Sub): return a - b
                if isinstance(node.op, ast.Mult): return a * b
                if isinstance(node.op, (ast.Div, ast.FloorDiv)) and b: return a // b
                if isinstance(node.op, ast.Mod) and b: return a % b
            raise ValueError('Unsupported integer expression')
        try:
            return walk(ast.parse(expression.strip(), mode='eval').body)
        except (SyntaxError, ZeroDivisionError) as exc:
            raise ValueError('Unsupported integer expression') from exc


def proven_threads(expressions):
    code = expressions.code
    launches = list(re.finditer(r'<<<\s*\w+\s*,\s*(\w+)\s*>>>', code))
    if len(launches) != 1:
        raise ValueError('Requires one standard 1D GEMM launch')
    name = launches[0][1]
    declarations = list(re.finditer(r'\bdim3\s+' + re.escape(name) + r'\s*\(([^;]+)\);', code))
    if len(declarations) != 1:
        raise ValueError('Ambiguous launch dimensions')
    parts = declarations[0][1].split(',')
    dims = [expressions.value(part, declarations[0].start()) for part in parts]
    if len(dims) not in (1, 3) or any(v != 1 for v in dims[1:]) or not 0 < dims[0] <= 1024:
        raise ValueError('Only proven one-dimensional launch geometry is supported')
    return dims[0]


def thread_bindings(threads, tid=0):
    return {'blockDim.x': threads, 'blockDim.y': 1, 'blockDim.z': 1,
            'threadIdx.x': tid, 'threadIdx.y': 0, 'threadIdx.z': 0}


def fixed_load_iterations(source):
    expressions = Expressions(source)
    threads = proven_threads(expressions)
    edits = []
    offset = 'scope_fixed_load_offset'
    if re.search(r'\b' + offset + r'\b', expressions.code):
        raise ValueError('Fixed-load loop already present')
    for match in LOAD_LOOP.finditer(expressions.code):
        variable, start, bound, stride = match.groups()
        end = loop_end(expressions.code, match)
        body = expressions.code[match.end():end]
        if 'float4' not in body or not re.search(r'\b(?:As|Bs)\[', body):
            continue
        if re.search(r'\b' + re.escape(variable) + r'\s*(?:\+\+|--|[+*/%-]?=(?!=))|(?:\+\+|--)\s*\b' + re.escape(variable) + r'\b', body):
            raise ValueError('Producer loop modifies its ownership index')
        size = expressions.value(bound, match.start(), thread_bindings(threads))
        if size <= 0 or size % threads or size > 1048576:
            raise ValueError('Incomplete or unsupported vector task groups')
        for tid in range(threads):
            bindings = thread_bindings(threads, tid)
            if (expressions.value(start, match.start(), bindings) != tid
                    or expressions.value(stride, match.start(), bindings) != threads
                    or expressions.value(bound, match.start(), bindings) != size):
                raise ValueError('Load ownership/stride is not uniform and complete')
        replacement = (f'for (int {offset} = 0; {offset} < {size}; {offset} += {threads}) {{\n'
                       f'    const int {variable} = {start} + {offset};')
        # Preserve any existing unroll pragma, rather than stacking pragmas.
        if not re.search(r'#pragma\s+unroll\s*$', expressions.code[:match.start()]):
            replacement = '#pragma unroll\n' + replacement
        edits.append((match.start(), match.end(), replacement))
    if len(edits) != 4:
        raise ValueError('Expected first/next A/B vector-load loops')
    for start, end, replacement in reversed(edits):
        source = source[:start] + replacement + source[end:]
    return source


def vector_shared_b(source):
    expressions = Expressions(source)
    declaration = re.search(r'__shared__\s+float\s+Bs\[(\d+)\]\[BK\]\[BN\];', expressions.code)
    if not declaration or int(declaration[1]) < 1 or expressions.tile.get('BN', 0) % 4:
        raise ValueError('Requires unpadded B with four-float row/stage strides')
    # Capture the entire four-element scatter. Each replacement is checked in its
    # own lexical scope, including aliases and inline BN/4 arithmetic.
    pattern = re.compile(r'Bs\[(?P<s>\w+)\]\[(?P<k>\w+)\]\[(?P<n>\w+)\s*\+\s*0\]\s*=\s*(?P<v>\w+)\.x;\s*'
                         r'Bs\[(?P=s)\]\[(?P=k)\]\[(?P=n)\s*\+\s*1\]\s*=\s*(?P=v)\.y;\s*'
                         r'Bs\[(?P=s)\]\[(?P=k)\]\[(?P=n)\s*\+\s*2\]\s*=\s*(?P=v)\.z;\s*'
                         r'Bs\[(?P=s)\]\[(?P=k)\]\[(?P=n)\s*\+\s*3\]\s*=\s*(?P=v)\.w;')
    edits = [(declaration.start(), declaration.end(),
              declaration[0].replace('__shared__', '__shared__ __align__(16)', 1))]
    total = expressions.tile['BK'] * expressions.tile['BN'] // 4
    if not 0 < total <= 65536:
        raise ValueError('Unsupported vector ownership domain')
    for match in pattern.finditer(expressions.code):
        # The supported producer loops enumerate v, covering exactly one B tile.
        loops = [loop for loop in LOAD_LOOP.finditer(expressions.code)
                 if loop.end() <= match.start() < loop_end(expressions.code, loop)]
        if not loops:
            raise ValueError('Unrecognized B producer loop')
        variable = loops[-1][1]
        for v in range(total):
            n = expressions.value(match['n'], match.start(), {variable: v})
            k = expressions.value(match['k'], match.start(), {variable: v})
            if n != (v % (expressions.tile['BN'] // 4)) * 4 or k != v // (expressions.tile['BN'] // 4):
                raise ValueError('B vector ownership/alignment not proven')
        edits.append((match.start(), match.end(),
                      f'FLOAT4(Bs[{match["s"]}][{match["k"]}][{match["n"]}]) = {match["v"]};'))
    if len(edits) != 3:
        raise ValueError('Expected complete first/next B scatter groups')
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source
