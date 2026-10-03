"""Replay a selection request without generating or compiling kernels."""
import argparse
import json
import time
from pathlib import Path

import yaml

from SCOPE.app import build_joint_tiling_candidates, make_diverse_tiling_pool, compact_joint_tiling_ir_summary
from SCOPE.llm.openai_client import OpenAICompatibleClient, extract_json_object
from SCOPE.llm.strategy_selector import build_joint_tiling_messages, validate_joint_tiling_response, DEFAULT_JOINT_TILING_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--reasoning-effort', choices=('low', 'high', 'max'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = yaml.safe_load((root / 'conf.yaml').read_text(encoding='utf-8'))
    client = OpenAICompatibleClient(config['llm'])
    read = lambda path: json.loads((root / path).read_text(encoding='utf-8'))
    ir = read('data/IRs/ir_patch/optir.extracted.json')
    library = read('data/lib/strategy_library.json')
    allowed = set(read('results/chain/chain_root.Tiling_BlockTileSelection.available.json')['available_strategy_ids'])
    limits = read('results/check/evolution_summary.json')['tiling_search_funnel']
    legal = [c for c in build_joint_tiling_candidates(library, ir) if c['block_strategy_id'] in allowed]
    pool = make_diverse_tiling_pool(legal, limits['resource_pool_size'])
    messages = build_joint_tiling_messages(compact_joint_tiling_ir_summary(ir), pool,
                                           limits['llm_top_n'], DEFAULT_JOINT_TILING_PROMPT)
    kwargs = client._completion_kwargs(messages, client.selection_timeout_seconds,
                args.max_tokens or client._selection_max_tokens,
                json_mode=True, thinking_mode=client._selection_thinking_mode,
                reasoning_effort=client._selection_reasoning_effort)
    if args.reasoning_effort:
        kwargs['extra_body'] = {**kwargs.get('extra_body', {}), 'reasoning_effort': args.reasoning_effort}
    report = {'model': kwargs['model'], 'max_tokens': kwargs['max_tokens'],
              'thinking': kwargs.get('extra_body'), 'legal_count': len(legal), 'pool_count': len(pool),
              'prompt_chars': sum(len(m['content']) for m in messages),
              'response_format': kwargs.get('response_format')}
    started = time.monotonic()
    try:
        response = client._client.with_options(max_retries=0).chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ''
        report.update(finish_reason=choice.finish_reason, content=content,
                      reasoning_chars=len(getattr(choice.message, 'reasoning_content', '') or ''),
                      usage=response.usage.model_dump() if response.usage else None)
        if content:
            try:
                report['validated_selection'] = validate_joint_tiling_response(
                    extract_json_object(content), pool, limits['llm_top_n'])
            except Exception as exc:
                report['validation_error'] = type(exc).__name__ + ': ' + str(exc)
        else:
            report['validation_error'] = 'Empty final content (production currently substitutes {})'
    except Exception as exc:
        report['request_error_type'] = type(exc).__name__
        report['request_status_code'] = getattr(exc, 'status_code', None)
    report['elapsed_seconds'] = round(time.monotonic() - started, 2)
    output = root / 'results/diagnostics' / ('tiling_response_' + time.strftime('%Y%m%d_%H%M%S') + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(output)


if __name__ == '__main__':
    main()
