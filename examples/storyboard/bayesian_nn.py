'''
To make a neural net bayesian we need to make the params into random vars, and guide them if we want VI.

I propose a new helper pyro.random_module to construct an (implicit) distribution over nns from a module and a matching prior.

prior(name, shape) must return an iterator over tensors, whose signatures matche the sizes of module.parameters().
the shape arg will be an iterator over parameter objects (ie the result of module.parameters()).

note that initializers could (should?) have same signature as priors.
'''

#To start, let's just write a non-bayesian neural net, with supervised training:
classify = nn.Sequential(nn.Conv2d(784,200,5), nn.ReLU(), nn.Conv2d(200,10,5), nn.ReLU())

def model(data):
    nn = pyro.module("classifier", classify)
    map_data(data, lambda i, d: pyro.observe("obs"+i, Categorical(nn.forward(d.data)), d.cll)

def guide(data):
    None

             
#now let's make it bayesian. we'll need the prior fn:
def prior(name, shape):
    for p in shape
        yield pyro.sample(name+uuid, Gaussian(p.size()))

#we use this as so:
classifier_dist = pyro.random_module("classifier", classify, prior) #make the module into an implicit distribution on nets
classifier = classifier_dist() #sample a random net
class_weights = classifier.forward(data) #use the net (as ordinary fn)

#note that in the corresponding guide (for VI) we'd want to add (implicitly or explicitly) params to the prior dist.
#here's a sketch of a bayesian nn with a layer-wise independent posterior:

def prior(name, shape):
    for p in shape
        yield pyro.sample(name+uuid, Gaussian(p.size()))

def posterior(name, shape):
    for p in shape
        weight_dist = Gaussian(pyro.param(name+"param"+uuid, p.size()))
        yield pyro.sample(name+uuid, weight_dist)

model(data):
    classifier_dist = pyro.random_module("classifier", classify, prior) #make the module into an implicit distribution on nets
    nn = classifier_dist() #sample a random net
    map_data(data, lambda i, d: pyro.observe("obs"+i, Categorical(nn.forward(d.data)), d.cll)
       
guide(data):
    classifier_dist = pyro.random_module("classifier", classify, posterior) #make the module into an implicit distribution on nets
    nn = classifier_dist() #sample a random net
             
             
#here's a sketch of the posterior function to use for a "deep posterior":
update = nn.rnn(...) #hm, assumes all weight matrices are same size.   
predict = nn.mlp(..)
def posterior(name, shape):
    update = pyro.module("update", update)
    predict = pyro.module("predict", predict)
    hidden = Variable(nn.ones(10))
    for p in shape
        weight_dist = Gaussian(predict.forward(hidden))
        weights = pyro.sample(name+uuid, weight_dist)
        hidden = update.forward(nn.concat(weights, hidden))
        yield weights           