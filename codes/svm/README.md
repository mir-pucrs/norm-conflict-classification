
# Norm Conflict Classification Execution

## Binary Classification

#### On [settings.conf](../settings.conf):
    BINARY=True
    MODE=offset or concat

   Run:

```sh
 python svm.py path_to/10-fold_binary.json path_to/conflicts.csv path_to/non_conflicts.csv
  ```

## Multiclass Classification


### TypeC + Non (E_{off})
####  On [settings.conf](../settings.conf):
    BINARY=False
    MODE=offset

Run:

```sh
python svm.py path_to/10-fold.json path_to/conflicts.csv path_to/non_conflicts.csv
```

### TypeC + Non (E_{conc})
#### On [settings.conf](../settings.conf):
        BINARY=False
        MODE=concat

Run:

```sh
python svm_conflict.py path_to/10-fold.json path_to/conflicts.csv path_to/non_conflicts.csv
```

### TypeC (E_{off})

   #### On [settings.conf](../settings.conf):
    BINARY=False
    MODE=offset

Run:

```sh
python svm_conflict.py path_to/10-fold.json path_to/conflicts.csv path_to/non_conflicts.csv
```

### TypeC (E_{conc})

   #### On [settings.conf](../settings.conf):
    BINARY=False
    MODE=concat

Run:

```sh
python svm_conflicts.py path_to/10-fold.json path_to/conflicts.csv path_to/non_conflicts.csv    
```
