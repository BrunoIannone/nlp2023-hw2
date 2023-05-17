from tqdm import tqdm
from datasets import load_metric

import time
def train(train_dataloader, epochs,optimizer,model,device,valid_dataloader,label_to_idx):
    number_of_training_steps = len([b for b in train_dataloader])
    for epoch in range(epochs):  # loop over the dataset multiple times
        batch_pbar = tqdm(enumerate(train_dataloader), total=number_of_training_steps)
        for i, batch in batch_pbar:
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            # move batch data to the specified device
            #print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            # ** operator converts batch items in named arguments, (e.g. 'input_ids', 'attention_mask_ids' ...), taken as input by the model forward pass
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            batch_pbar.set_postfix({'loss': loss.item()})
        validation(valid_dataloader,device,model,label_to_idx)

def validation(val_dataloader,device,model,label_to_idx):
    # load_metric is a huggingface utility to load the most-used metrics in ML.
    metric = load_metric("seqeval")

    # predict
    predictions, labels = [], []
    for batch in tqdm(val_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, compute_predictions=True)
        y_true = batch["labels"].tolist()
        y_pred = outputs["predictions"].tolist()
        label_list = label_to_idx.key()
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        predictions += true_predictions
        labels += true_labels

    # overall score print
    scores = metric.compute(predictions=predictions, references=labels)
    for k, v in scores.items():
        print(f"{k}: {v}")