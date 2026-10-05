"""FALAT stage modules, one per paper stage. Each exposes ``run(runner, traj, trace, ctx)``.

    stage0_task_spec               (assumption) answer_type / constraints
    stage1_prior                   external prior π                     §2.1, A.1
    stage1_hierarchy               M per k-window, candidate set C, R   §2.1, A.1
    stage2_typed_dependency        typed transitions, C -> C′           §2.2, A.2
    stage3a_candidate_selection    C′ -> C_H                            §2.3, A.3
    stage3b_candidate_verification C_H -> t̂                              §2.3, A.3
    stage4_local_research          t̂ -> t̃                                §2.4, A.4
"""
