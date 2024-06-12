from datasets import load_dataset
import random
from PIL import ImageDraw, ImageFont, Image
from transformers import ViTImageProcessor
import torch
import numpy as np
import evaluate
from datasets import load_metric
from transformers import ViTForImageClassification, ViTModel, ViTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers import TrainingArguments
from data import create_dataloader, get_dataset
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from models.cnnDetection.validate import CNNmethod
from models.selfblended.validate import SelfBlendedMethod
from models.universalFake.validate import UniversalFakeMethod
import torch.multiprocessing as mp


class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=3):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None

if __name__ == '__main__':
    EPOCHS = 3
    BATCH_SIZE = 64
    LEARNING_RATE = 2e-5
    mp.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CNNmethod.load_model(device)
    SelfBlendedMethod.load_model(device)
    UniversalFakeMethod.load_model(device)
    
    train_set = get_dataset()
    data_loader = create_dataloader(train_set)
    test_set = get_dataset('datasets/test/progan/')
    test_loader  = create_dataloader(test_set)
    model = ViTForImageClassification(2).to(device)    
    # Feature Extractor
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cross Entropy Loss
    loss_func = nn.CrossEntropyLoss()
    # Use GPU if available  
    
# if False: 
    for epoch in range(EPOCHS):        
        for step, (x, y) in enumerate(data_loader):
            print("Train process",x, y)
            # Change input array into list with each batch being one element
            x = np.split(np.squeeze(np.array(x)), BATCH_SIZE)
            # Remove unecessary dimension
            for index, array in enumerate(x):
                x[index] = np.squeeze(array)
            # Apply feature extractor, stack back into 1 tensor and then convert to tensor
            x = torch.tensor(np.stack(processor(x)['pixel_values'], axis=0))
            # Send to GPU if available
            x, y  = x.to(device), y.to(device)
            b_x = Variable(x)   # batch x (image)
            b_y = Variable(y)   # batch y (target)
            # Feed through model
            output, loss = model(b_x, None)
            # Calculate loss
            if loss is None: 
                loss = loss_func(output, b_y)   
                optimizer.zero_grad()           
                loss.backward()                 
                optimizer.step()

            if step % 50 == 0:
                # Get the next batch for testing purposes
                test = next(iter(test_loader))
                test_x = test[0]
                # Reshape and get feature matrices as needed
                test_x = np.split(np.squeeze(np.array(test_x)), BATCH_SIZE)
                for index, array in enumerate(test_x):
                    test_x[index] = np.squeeze(array)
                test_x = torch.tensor(np.stack(processor(test_x)['pixel_values'], axis=0))
                # Send to appropirate computing device
                test_x = test_x.to(device)
                test_y = test[1].to(device)
                # Get output (+ respective class) and compare to target
                test_output, loss = model(test_x, test_y)
                test_output = test_output.argmax(1)
                # Calculate Accuracy
                accuracy = (test_output == test_y).sum().item() / BATCH_SIZE
                print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy)

# ds = load_dataset('beans')
# print(ds)

# example = ds['train'][400]
# print(example)

# image = example['image']
# print(image)

# labels = ds['train'].features['labels']
# print(labels)

# labels.int2str(example['labels'])

def show_examples(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):

    w, h = size
    labels = ds['train'].features['labels'].names
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)
    # font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 24)

    for label_id, label in enumerate(labels):

        # Filter the dataset by a single label, shuffle it, and grab a few samples
        ds_slice = ds['train'].filter(lambda ex: ex['labels'] == label_id).shuffle(seed).select(range(examples_per_class))

        # Plot this label's examples along a row
        for i, example in enumerate(ds_slice):
            image = example['image']
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            draw.text(box, label, (255, 255, 255), font=font)

    return grid

# show_examples(ds, seed=random.randint(0, 1337), examples_per_class=3)

# model_name_or_path = 'google/vit-base-patch16-224-in21k'
# processor = ViTImageProcessor.from_pretrained(model_name_or_path)
# processor(image, return_tensors='pt')

# def transform(example_batch):
#     # Take a list of PIL images and turn them to pixel values
#     inputs = processor([x for x in example_batch['image']], return_tensors='pt')

#     # Don't forget to include the labels!
#     inputs['labels'] = example_batch['labels']
#     return inputs

# prepared_ds = ds.with_transform(transform)
# def collate_fn(batch):
#     return {
#         'pixel_values': torch.stack([x['pixel_values']for x in batch]),
#         'labels': torch.tensor([x['labels'] for x in batch])
#     }
    
# metric = evaluate.load("accuracy")
# def compute_metrics(p):
#     return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# model = ViTForImageClassification.from_pretrained(
#     model_name_or_path,
#     num_labels=3,
#     id2label={str(i): c for i, c in enumerate(labels.names)},
#     label2id={c: str(i) for i, c in enumerate(labels.names)}
# )
# # model.to(device)



# training_args = TrainingArguments(
#     output_dir="./vit-base-search",
#     per_device_train_batch_size=16,
#     evaluation_strategy="steps",
#     num_train_epochs=4,
#     fp16=True,
#     save_steps=100,
#     eval_steps=100,
#     logging_steps=10,
#     learning_rate=2e-4,
#     save_total_limit=2,
#     remove_unused_columns=False,
#     push_to_hub=False,
#     report_to='tensorboard',
#     load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=model.to(device),
#     args=training_args,
#     data_collator=collate_fn,
#     compute_metrics=compute_metrics,
#     train_dataset=prepared_ds["train"],
#     eval_dataset=prepared_ds["validation"],
#     tokenizer=processor,
# )

# train_results = trainer.train()
# trainer.save_model()
# trainer.log_metrics("train", train_results.metrics)
# trainer.save_metrics("train", train_results.metrics)
# trainer.save_state()

# metrics = trainer.evaluate(prepared_ds['validation'])
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)

