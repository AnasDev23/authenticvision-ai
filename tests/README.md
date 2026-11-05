Run unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

These tests validate metric calculations and verify that a held-out generator is excluded from the training indices. They do not train the full neural network or validate real-world accuracy.

The updated splitter tests also verify that deployment mode includes Midjourney in training and keeps image indices disjoint between validation and test.
