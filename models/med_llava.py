datasets_path = 'eepy/datasets'
hf_cache_path = 'eepy/hf-cache'

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, AutoTokenizer, MistralForCausalLM, AdamW

#NOTE: Problem from line 126

cache_dir = "/mount/studenten-temp1/users/takamo/llava-med-v1.5-mistral-7b"

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### PARAMS to set ###
MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_CLASSES = 10 # for output layer 
LEARNING_RATE = 1e-4
EPOCHS = 3
DATASET = "vqarad"
# Llava_med_EPOCHS_LR_BATCH_DATASET
SAVED_MODEL_NAME = f"Llava_med_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}_{DATASET}.pth"


### MODEL and TOKENIZER ###
# Query tokenizer and model from cache
tokenizer = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True)
#model = LlamaForCausalLM.from_pretrained(cache_dir, local_files_only=True, device_map="auto")
model = MistralForCausalLM.from_pretrained(cache_dir, local_files_only=True, device_map="auto")

print(f"*** Finished querying tokenizer and llava-med model. ***\n\n*** Prepping data. ***")


### DATASET PROCESSING - VQA_RAD ###
# Format --> {'image': PIL.JPEG, 'question': 'are regions of the brain infarcted?', 'answer': 'yes'}
vqa_rad = load_dataset("flaviagiammarino/vqa-rad", cache_dir=cache_dir)

# answer_map format --> {'yes': 0, 'no': 1, 'cardiovascular': 2, 'right': 1984, ...}
answer_map = {}
for count, (train_item, test_item) in enumerate(zip(vqa_rad['train'], vqa_rad['test'])):
    train_answer = train_item.get('answer') 
    test_answer = test_item.get('answer')    

    if train_answer not in answer_map:
        answer_map[train_answer] = count 
    
    # if test_answer is not in the map, assign a unique integer index
    if test_answer not in answer_map:
        answer_map[test_answer] = count + len(vqa_rad['train']) 
    
answer_map["unknown"] = -1

        
def preprocess_data(data, max_length=MAX_LENGTH, answer_map=answer_map):
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    # Loop over each sample in batch
    for sample in data['question']:
        # Tokenize the question text
        encoding = tokenizer(
            sample,
            truncation=True,
            padding="max_length",
            max_length=max_length, 
        )
        
        # list --> ['yes', 'no', ...]
        answers = data["answer"]
        labels = []

        for answer in answers:
            # Map answers to int labels
            if answer not in answer_map:
                label = answer_map["unknown"]  # -1 for unknown answers
            else:
                label = answer_map[answer]

            labels.append(label)  # Add the label for each answer

        # Append results to batch lists
        input_ids_batch.append(encoding["input_ids"])
        attention_mask_batch.append(encoding["attention_mask"])
        labels_batch.append(labels)
    
    # batch results
    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch
    }

# Split TRAIN and TEST data & encode
train_vqarad = vqa_rad['train'].map(preprocess_data, batched=True) #, fn_kwargs={'answer_map': answer_map})
test_vqarad = vqa_rad['test'].map(preprocess_data, batched=True)

# format for pytorch --> input_ids, attention_mask, labels
train_vqarad.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_vqarad.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# dataloader
train_loader = DataLoader(train_vqarad, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_vqarad, batch_size=BATCH_SIZE, shuffle=False)


### PREP OUTPUT LAYER TO TRAIN ###
# freeze pre-trained model
for param in model.parameters():
    param.requires_grad = False

print(f"\n*** Defining output layer. ***")
# params for output layer
hidden_size = model.config.hidden_size

# output layer on top
output_layer = nn.Linear(hidden_size, NUM_CLASSES).to(device)

# loss and optimizer for output layer - adjust lr
optimizer = AdamW(output_layer.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()
# WORKS TIL HERE


# FROM HERE: PROBLEM (sth with padding and batches??)
### TRAINING ###
print(f"\n*** Start TRAINING ***")
for epoch in range(EPOCHS):
    model.eval()  # ensure the model stays in evaluation mode
    output_layer.train()  # only output layer in training mode


    for batch in train_loader:
        optimizer.zero_grad()

        # Get inputs from batch
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        with torch.no_grad():  # Freeze pre-trained model
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states[-1]  # last hidden layer

        # Pass the last token's hidden state through the output layer
        logits = output_layer(hidden_states[:, -1, :])
        loss = loss_fn(logits, labels)


        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    

# save model after last epoch   
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'output_layer_state_dict': output_layer.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, SAVED_MODEL_NAME)


"""
# for later: to call model
checkpoint = torch.load(SAVED_MODEL_NAME)
model.load_state_dict(checkpoint['model_state_dict'])
output_layer.load_state_dict(checkpoint['output_layer_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
"""