"""
This is purely for learning purposes. 
Reference used: https://arxiv.org/abs/2001.09186. 

m --> (s, t) : the compressed message
s --> an int in the range [2^(s_prec - t_prec), 2^s_prec]
t --> an immutable stack, has ints in the range [0, 2^t_prec)

The precisions must satisfy t_prec < s_prec
where 
t_prec --> precision of t
s_prec --> precision of s
p_prec --> precision of p

x --> symbol
c --> beginning of the interval corresponding to x
  --> is analogous to the cumulative 
p --> width of the interval corresponding to x
  --> P(x) == p / (2^p_prec)

The user provides the model over the symbols, in the following 
format:

(f, g, p_prec) 
where

f : s_bar -> x, (c, p) # maps s_bar to the symbol and subinterval corresponding to x
g : x -> c, p  # maps the symbol to a start point on, and width

think of s_bar as the latent variable. based on s_bar, we know that the symbol
is the x corresponding to the interval in which s_bar falls. 


"""

s_prec = 64
t_prec = 32
t_mask = (1 << t_prec) - 1
s_min  = 1 << s_prec - t_prec
s_max  = 1 << s_prec

#        s    , t
m_init = s_min, ()  # Shortest possible message

def rans(model):
    f, g, p_prec = model
    def push(m, x):
        s, t = m
        c, p = g(x)
        # Invert renorm
        while s >= p << s_prec - p_prec:
            s, t = s >> t_prec, (s & t_mask, t)
        # Invert d
        s = (s // p << p_prec) + s % p + c
        assert s_min <= s < s_max
        return s, t

    def pop(m):
        s, t = m
        # d(s)
        s_bar = s & ((1 << p_prec) - 1)
        x, (c, p) = f(s_bar)
        s = p * (s >> p_prec) + s_bar - c
        # Renormalize
        while s < s_min:
            t_top, t = t
            s = (s << t_prec) + t_top
        assert s_min <= s < s_max
        return (s, t), x
    return push, pop

def flatten_stack(t):
    flat = []
    while t:
        t_top, t = t
        flat.append(t_top)
    return flat

def unflatten_stack(flat):
    t = ()
    for t_top in reversed(flat):
        t = t_top, t
    return t


if __name__ == "__main__":
    import math

    log = math.log2

    # We encode some data using the example model in the paper and verify the
    # inequality in equation (20).

    # First setup the model
    p_prec = 3

    # Cumulative probabilities
    cs = {'a': 0,
          'b': 1,
          'c': 3,
          'd': 6}

    # Probability weights, must sum to 2 ** p_prec
    ps = {'a': 1,
          'b': 2,
          'c': 3,
          'd': 2}

    # Backwards mapping
    s_bar_to_x = {0: 'a',
                  1: 'b', 2: 'b',
                  3: 'c', 4: 'c', 5: 'c',
                  6: 'd', 7: 'd'}

    def f(s_bar):
        x = s_bar_to_x[s_bar]
        c, p = cs[x], ps[x]
        return x, (c, p)

    def g(x):
        return cs[x], ps[x]

    model = f, g, p_prec

    push, pop = rans(model)

    # Some data to compress
    xs = ['a', 'b', 'b', 'c', 'b', 'c', 'd', 'c', 'c']
    print("The sequence of symbols is: {}".format(xs))

    # Compute h(xs):
    h = sum(map(lambda x: log(2 ** p_prec / ps[x]), xs))
    print('Information content of sequence: h(xs) = {:.2f} bits.'.format(h))
    print()

    # Initialize the message
    m = m_init

    # Encode the data
    for x in xs:
        m = push(m, x)

    # Verify the inequality in eq (20)
    eps = log(1 / (1 - 2 ** -(s_prec - p_prec - t_prec)))
    print('eps = {:.2e}'.format(eps))
    print()

    s, t = m
    lhs = log(s) + t_prec * len(flatten_stack(t)) - log(s_min)
    rhs = h + len(xs) * eps
    print('Eq (20) inequality, rhs - lhs == {:.2e}'.format(rhs - lhs))
    print()

    # Decode the message, check that the decoded data matches original
    xs_decoded = []
    for _ in range(len(xs)):
        m, x = pop(m)
        xs_decoded.append(x)

    xs_decoded = list(reversed(xs_decoded))

    for x_orig, x_new in zip(xs, xs_decoded):
        assert x_orig == x_new

    # Check that the message has been returned to its original state
    assert m == m_init
    print('Decode successful!')
    print("Decoded message is: {}".format(xs_decoded))