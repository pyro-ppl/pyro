import math
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints
from pyro.infer import SVI, Predictive, TraceEnum_ELBO
from tqdm import trange
import requests
import tarfile
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer


def model(data, vocab_size, num_docs, num_topics, doc_idx=None):
    # Globals.
    eta = data.new_ones(vocab_size)
    with pyro.plate("topics", num_topics):
        beta = pyro.sample("beta", dist.Dirichlet(eta))

    # Locals.
    with pyro.plate("documents", data.shape[1]):
        alpha = data.new_ones(num_topics)
        theta = pyro.sample("theta", dist.Dirichlet(alpha))

        with pyro.plate("words", data.shape[0]):
            zeta = pyro.sample("zeta", dist.Categorical(theta))
            pyro.sample("doc_words", dist.Categorical(beta[..., zeta, :]),
                        obs=data)


def guide(data, vocab_size, num_docs, num_topics, doc_idx=None):
    # Parameters
    lambda_ = pyro.param("lambda", data.new_ones(num_topics, vocab_size))
    gamma = pyro.param("gamma", data.new_ones(num_docs, num_topics))
    phi = pyro.param("phi", data.new_ones(num_docs, data.shape[0], num_topics),
                     constraint=constraints.positive)
    phi = phi / phi.sum(dim=2, keepdim=True)  # Enforces probability

    # Topics
    with pyro.plate("topics", num_topics):
        pyro.sample("beta", dist.Dirichlet(lambda_))

    # Documents
    with pyro.plate("documents", data.shape[1]):
        pyro.sample("theta", dist.Dirichlet(gamma[..., doc_idx, :]))

        # Words
        with pyro.plate("words", data.shape[0]):
            pyro.sample(
                "zeta",
                dist.Categorical(phi[..., doc_idx, :, :].transpose(1, 0))
            )


def train(docs, vocab_size, num_topics, batch_size, learning_rate, num_epochs):
    # clear param store
    pyro.clear_param_store()

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO(num_particles=1))
    num_batches = int(math.ceil(docs.shape[0] / batch_size))

    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            idx = torch.arange(i * batch_size,
                               min((i + 1) * batch_size, len(docs)))
            batch_docs = docs[idx, :]
            loss = svi.step(batch_docs.T, vocab_size,
                            docs.shape[0], num_topics, idx)
            running_loss += loss

        epoch_loss = running_loss / docs.shape[0]
        bar.set_postfix(epoch_loss='{:.2f}'.format(epoch_loss))


def get_data(target_path, force_rewrite=False):
    # Download David Blei's AP dataset
    if not force_rewrite and not (Path(target_path) / 'ap.tgz').exists():
        url = "http://www.cs.columbia.edu/~blei/lda-c/ap.tgz"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with (Path(target_path) / 'ap.tgz').open('wb') as f:
                f.write(response.raw.read())

        # Untar
        tar = tarfile.open(Path(target_path) / 'ap.tgz', "r:gz")
        tar.extractall(path=target_path)
        tar.close()

    # Load vocabulary in a dataframe
    with (Path(target_path) / 'ap/vocab.txt').open('r') as f:
        vocab = [x.strip() for x in f.readlines()]

    vocab = pd.DataFrame(columns=['word'], data=vocab)
    reserved = pd.DataFrame(columns=['word'],
                            data=['blank, reserved to padding'])
    vocab = reserved.append(vocab, ignore_index=True)
    vocab['index'] = vocab.index

    # Load documents
    if not force_rewrite and (Path(target_path) / 'ap/docs.pt').exists():
        docs = torch.load(Path(target_path) / 'ap/docs.pt')
        return docs, vocab

    with (Path(target_path) / 'ap/ap.txt').open('r') as f:
        docs = [x.strip() for x in f.readlines() if not x.strip().startswith('<')]

    # Tokenize docs
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()
        docs[idx] = tokenizer.tokenize(docs[idx])

    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    # Lemmatize docs
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Substitute words by their indexes
    for doc_id in range(len(docs)):
        df = pd.DataFrame(columns=['word'], data=docs[doc_id])
        df = pd.merge(df, vocab[['index', 'word']], how='left', on='word').dropna()
        docs[doc_id] = torch.from_numpy(df['index'].astype(int).values)

    # Remove docs with zero length (2 occurrences) and pad docs with 0 index
    docs = [doc for doc in docs if len(doc) > 0]
    docs = pad_sequence(docs, batch_first=True, padding_value=0)
    torch.save(docs.short(), Path(target_path) / 'ap/docs.pt')

    return docs, vocab


def print_top_topic_words(docs, vocab_size, num_topics, vocab):
    predictive = Predictive(model, guide=guide, num_samples=100,
                            return_sites=["beta", 'obs'])

    i = 0
    batch_size = 32
    idx = torch.arange(i * batch_size,
                       min((i + 1) * batch_size, len(docs))).cpu()
    batch_docs = docs[idx, :].cpu()
    samples = predictive(batch_docs.T, vocab_size,
                         docs.shape[0], num_topics, idx)

    beta = samples['beta'].mean(dim=0).squeeze().detach().cpu()

    for i in range(beta.shape[0]):
        sorted_, indices = torch.sort(beta[i], descending=True)
        df = pd.DataFrame(indices[:20].numpy(), columns=['index'])
        print(pd.merge(df, vocab[['index', 'word']],
                       how='left', on='index')['word'].values)
        print()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    docs, vocab = get_data('/Users/carlossouza/Downloads')
    print(f'Data loaded: {docs.shape[0]} documents, {docs.shape[1]} words/doc, '
          f'{len(vocab)} vocabulary size.')

    docs = docs.float().to(device)
    vocab_size = len(vocab)
    num_topics = 20

    train(docs, vocab_size, num_topics, 32, 1e-3, 50)
    print_top_topic_words(docs, vocab_size, num_topics, vocab)

