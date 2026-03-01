import csv

class MasterTable:
    def __init__(self):
        self.rows = []

    def add(self, model, depth, activation, optimizer,
            params, train_acc, val_acc, test_acc):

        self.rows.append({
            "Model": model,
            "Depth": depth,
            "Activation": activation,
            "Optimizer": optimizer,
            "Parameters": params,
            "Train Acc": round(train_acc,4),
            "Val Acc": round(val_acc,4),
            "Test Acc": round(test_acc,4)
        })

    def display(self):
        print("\n========== MASTER RESULT TABLE ==========")
        for r in self.rows:
            print(r)

    def save(self, filename="master_table.csv"):
        keys = self.rows[0].keys()
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)