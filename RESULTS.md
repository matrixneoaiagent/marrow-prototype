# Marrow results

Marrow prototype - Listing 65 self-test
========================================================================
Fake data only. No real movement, text, audio or camera data.
Styles: pointer_keyboard, slow_pointer_dictation, switch_onscreen
Sessions: 372  (train/validation/test, owner/alternate/mimic)

Threshold (fixed on validation, before any test score)
  rule: lowest error trade-off: maximise TPR subject to validation FAR <= 0.05; ties go to the higher threshold
  target FAR: 5.0%
  locked threshold: 0.459481
  validation FAR/FRR: 1.56% / 0.00%
  lock file: artifacts/threshold_lock.json

Held-out test results
  owner FRR:    5.56%  (2/36, 95% CI 1.5%-18.1%)
  stranger FAR: 2.78%  (2/72)
  mimic FAR:    6.25%  (3/48)
  combined FAR: 4.17%  (5/120, 95% CI 1.8%-9.4%)
  ROC AUC:      0.979
  swap test:    0/36 owner sessions opened by a profile trained on the same-style other person
  swap, raw:    1/36 opened before the fixed 0.10 safety margin (diagnostic only; the margin is part of the locked rule)
  cross-style:  0/144 accepted (reported separately; a profile is enrolled per input style)

Per input style
  style                          FRR   tired  stranger   mimic    swap
  pointer_keyboard             8.33%  16.67%     0.00%   6.25%   0.00%
  slow_pointer_dictation       8.33%  16.67%     8.33%   6.25%   0.00%
  switch_onscreen              0.00%   0.00%     0.00%   6.25%   0.00%

Per owner
  alice    (pointer_keyboard): owner 11/12 accepted, stranger 0/24 accepted, mimic 1/16 accepted, swap 0/12 accepted
  nia      (switch_onscreen): owner 12/12 accepted, stranger 0/24 accepted, mimic 1/16 accepted, swap 0/12 accepted
  ravi     (slow_pointer_dictation): owner 11/12 accepted, stranger 2/24 accepted, mimic 1/16 accepted, swap 0/12 accepted

Five conditions
  [PASS] 1_open_source_one_command
  [PASS] 2_fake_data_two_styles_with_roles
  [PASS] 3_swap_test_rejects_owner
  [PASS] 4_threshold_fixed_and_rates_reported
  [PASS] 5_plain_format_offline_train_test

Pre-registered targets (reported, not part of the five conditions)
  [PASS] far_le_5_percent
  [PASS] frr_le_20_percent
  [PASS] mimic_far_le_15_percent

Elapsed: 49.82s. Offline: yes.
Verdict: all five conditions pass; pre-registered targets pass

The machine-readable form is [`artifacts/results.json`](artifacts/results.json); the threshold lock is [`artifacts/threshold_lock.json`](artifacts/threshold_lock.json).
