import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
#import sklearn
#from sklearn.model_selection import train_test_split

torch.manual_seed(1235)

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx) # (B,T,C)

        B, T, C, = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss


def preprocess(text):

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    #print(''.join(chars))    
    print(vocab_size)

    #create a mapping from charachters to integers
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}
    #enc = tiktoken.get_encoding('gpt2')
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    #print(encode("Sir Arthur"))
    #print(decode(encode("Sir Arthur")))
    #print("Tested")

    #encode the entire text dataset and stor it into a toch.Tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data[:1000])

    #Split up the data into train and validation sets
    n= int(0.9 * len(data))
    train_data= data[:n]
    val_data = data[n:]
    #X_train, X_test, y_train, y_test = train_test_split(train, test,
    #test_size=0.2, shuffle = True, random_state = 8)

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    #test_size=0.25, random_state= 8) # 0.25 x 0.8 = 0.2

    block_size= 8
    train_data[:block_size+1]

    x = train_data[:block_size]
    y = train_data[1:block_size + 1 ]
    for t in range(block_size):
        context = x [:t+1]
        target = y[t]
        print(f"when input is {context} the target: {target}")

    torch.manual_seed(1235)
    batch_size = 4
    block_size = 8

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+ block_size] for i in ix])
        y = torch.stack([data[i +1 :i+ block_size + 1 ] for i in ix])
        return x,y
    xb, yb = get_batch("train")
    print("inputs: ")
    print(xb.shape)
    print(xb)
    print("targets:" )
    print(yb.shape)
    print(yb)
    
    print("--------")

    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b,t]
            print(f"when input is {context.tolist()} the target: {target}")

    print(xb)

    m = BigramLanguageModel(vocab_size)
    logits, loss = m(xb,yb)
    print(logits.shape)
    print(loss)

    print("Done")

    

def read_corpus():

    # with open ('wiki.tr.txt', 'r', encoding='utf-8') as f:
    #     text = f.read()

    with open ('tr_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    #print(("Length of the dataset in charachters: ", len(text) ))
    #print(text[:1000])
    #print(len(text))
    n= int(0.1 * len(text))
    preprocess(text[:n])

def main():
    read_corpus()


if __name__ == '__main__':
    main()