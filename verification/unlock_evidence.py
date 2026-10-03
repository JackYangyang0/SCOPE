"""Label evidence without suppressing findings or asserting correctness proofs."""
import copy


def classify_diagnosis(diagnosis, summary):
    result = copy.deepcopy(diagnosis or {})
    runtime_pass = all(summary.get(key) == 'pass' for key in
                       ('compile_status', 'correctness_status', 'runtime_safety_status'))
    for defect in result.get('defects', []) or []:
        evidence = defect.get('evidence') or {}
        message = str(evidence.get('message', '')).lower()
        proof_gap = any(term in message for term in ('unproven', 'lacks alignment guard/proof', 'missing proof'))
        defect['selection_evidence_kind'] = (
            'static_proof_gap' if proof_gap else 'reported_defect_requires_inspection')
        defect['tested_execution_passed'] = runtime_pass
        if proof_gap:
            defect['selection_guidance'] = ('Inspect current source and refresh proof metadata. '
                'This finding alone is not evidence of an observed runtime failure; passing tests '
                'also do not prove correctness for untested inputs.')
    return result
