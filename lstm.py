import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, input_size, embedding_length, hidden_size, output_size, num_layers, bidirectional, dropout):
		super(LSTMClassifier, self).__init__()
		
		self.batch_size = batch_size
		self.input_size = input_size
		self.embedding_length = embedding_length
		self.hidden_size = hidden_size
		self.output_size = output_size
		
		
		self.word_embeddings = nn.Embedding(input_size, embedding_length)
		self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
		self.label = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)
		
		
	def forward(self, input_sentence, batch_size=None):

		input = self.word_embeddings(input_sentence)
		input = input.permute(1, 0, 2)

		if batch_size is None:
			h_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).cuda())
		else:
			h_0 = Variable(torch.zeros(4, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(4, batch_size, self.hidden_size).cuda())

		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

		final_output = self.label(final_hidden_state[-1])
		final_output = self.softmax(final_output)
		
		return final_output