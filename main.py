import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score

# imported packages correctly?
print("successfully!")

# Load the data from the CSV file
data = pd.read_csv('/datasets/summaries/data.csv')
print("Read file")

# sample only 20% of the data
data = data.sample(frac=0.1)
print("Got sample")

# drop missing values
data.dropna(inplace=True)
print("Dropped empty values")

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)
print("split the data sets")

# Extract the summaries and full stories from the training data
train_summaries = train_data['Summary']
train_stories = train_data['Content']
print("extracted specific columns from training data")

# Extract the summaries and full stories from the testing data
test_summaries = test_data['Summary']
test_stories = test_data['Content']
print("extracted specific columns from testing data")

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("created the tokenizer")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
print("defined the model")

# Convert the data into input format for the model
train_inputs = tokenizer.batch_encode_plus(train_stories.tolist(), max_length=512, pad_to_max_length=True, return_tensors='pt')
test_inputs = tokenizer.batch_encode_plus(test_stories.tolist(), max_length=512, pad_to_max_length=True, return_tensors='pt')
print("tokenizied the training and the testing data")

# Train the model on the training data
model.train()
print("trained the model")
model.zero_grad()
print("set all the gradients of the model's parameters to zero")
outputs = model(input_ids=train_inputs['input_ids'], labels=train_summaries)
print("defined the outputs")
loss, logits = outputs[:2]
loss.backward()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
optimizer.step()

# Test the model on the testing data
model.eval()
outputs = model(input_ids=test_inputs['input_ids'])
predictions = outputs[0]

# Calculate the accuracy of the model
accuracy = accuracy_score(test_summaries, predictions)
print('Accuracy:', accuracy)


