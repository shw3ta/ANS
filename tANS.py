'''
Implementation of the tabled variant of Asymmetric Numeral Systems
following the algorithm in Duda et. al, "Compcrypt -- Lightweight ANS
-based Compression and Encryption "

Here, ANS compression is treated as a triplet of algorithms <I, C, D> where
I : Initialization
C : Compress/encode
D : Decompress/decode

Tables for C, D stages are set up in I. 

This is a naive implementation to serve as a proof of concept for the example given in the paper.
'''

# imports
from math import log, floor

############################
##     INITIALIZATION     ##
############################

# defining symbols and frequencies
# in practice, we define a function that builds a p.d on the go and feeds it in as below
f_s = {"0":3, "1":8, "2":5}
cumul = sum(f_s.values())
R = int(log(cumul, 2)) # table parameter;
L = 1 << R # table size

# q = {L_0, L_1, L_2} where L_i = L*p_i
q_s = {s: L*f//cumul for s, f in f_s.items()} # our approximate quantized frequencies
state_labels = [l for l in range(L, 2*L)]
state_labels_s = {s : [l for l in range(f, 2*f)] for s, f in q_s.items()}

print(f"\nR (table param or log size) : {R}\nL (table size or cardinality of set of state labels) : {L}")
print(f"\nthe quantized symbol frequencies : {q_s}")
print(f"\nstate labels : {state_labels}")
print(f"\nstate labels for each symbol : {state_labels_s}")

# the unique values in the state labels per symbols are made into a set,
# and the encoding function is built as a table on top of that. to do this, 
# we first need to use a symbol spread function. 
all_pre_labels = []
for symbol, l in state_labels_s.items():
    all_pre_labels += [(i, symbol) for i in l]

print(f"\ncollated labels : {all_pre_labels}")

# spread symbols using rule: (i + 4 * (i mod 4)) mod 16 (for this case)
# does (i + R*(i mod R)) mod L work as a general rule?
spread = [0] * L
for i in range(L):
    index = (i + 4 * (i % 4)) % L
    # print(index)
    spread[index] = all_pre_labels[i]

print(f"\nthe spread : {spread}")

# scrambling: shift within each R sized block by a number given by some crypto key
# for now, key is set to '2130' to get verifiable outputs with Duda's example
spread_scrambled = [None]*L
key = "2222111133330000"
for i in range(L):
    block = i//R
    inblock_shift = int(key[i])
    inblock_index = (i  - inblock_shift) % 4
    index = 4 * block + inblock_index
    spread_scrambled[i] = spread[index]   
        
print(f"\nscrambled: {spread_scrambled}") 

# building the decoding table and function
D = {k: None for k in state_labels}
ctr = {s: 0 for s in q_s.keys()} # symbol specific counters

# D(x) = (y, s) tuples where x = state_label
for state_label in D.keys():
    D[state_label] = {}
    s = spread_scrambled[state_label - L][1] # the corresponding symbol    
    symbol_list = state_labels_s[s]

    D[state_label]['y'] = symbol_list[ctr[s]]
    D[state_label]['s'] = s
    D[state_label]['k'] = R - int(log(D[state_label]['y'], 2)) # num bits to read out of stream
    D[state_label]['x_'] = D[state_label]['y'] << (D[state_label]['k']) # next state to go to is x_ + decimal(k extracted msbits)
    ctr[s] += 1
    
print(f"\ndecoding table (reassigned): {D[28]}") 

# enumerating the encoding function C(s, y) = x
l = set([k[0] for k in all_pre_labels])
inverse_map = {(d['s'], d['y'] ) : k for k, d in D.items()}
C = {k : {s: None for s in q_s.keys()} for k in l}
for k in C.keys():
    for t in inverse_map.keys():
        if t[1] == k:
            s = t[0]
            C[k][s] = inverse_map[t]

# print(f"\ninverse map: {inverse_map} len of inverse_map: {len(inverse_map)}")
# print(f"\nencoding function: {C} len of encoding function: {len(C)}")

E = {k : {s: None for s in q_s.keys()} for k in state_labels}
for x in E.keys():
    for s in q_s.keys():
        L_s = len(state_labels_s[s])
        k = int((log((x/L_s), 2)))
        y = floor(x/(1 << k))
        bits = bin(int(x % (1 << k)))[2:].zfill(k)        
        E[x][s] = (C[y][s], bits) # this is where the encoding function is useful
    
print(f"\nencoding table : {E}")


######################
##     ENCODING     ##
######################

def Encoder(stream, x, E = E):
    '''
    Input: 
    stream : symbol stream (str) of len = n
    E   : encoding table 
    x   : initial state, belongs to state_labels

    Output:
    B   : bitstream (str)
    x   : the final state (int)
    for posterity, all intermediate variables are shown here, while
    they have already been computed in the table and needs just be used
    '''
    n = len(stream)
    B = ""
    x = x
    for i in range(n):
        s = stream[i]
        x, bits = E[x][s]
        print(f"\n bits produced in round {i} : {bits}")
        B += bits
    return B, x


######################
##     DECODING     ##
######################

def Decoder(B, x_f, D = D):
    '''
    Input: 
    B   : bitstream (str)
    D   : decoding table
    x_f : final state from encoding

    Output:
    stream  : symbol stream (str)
    '''
    x = x_f
    B = B[:: -1]
    print(f"\nreversed bitstream: {B}")
    stream = ""
    while len(B) > 0:
        y, s, k, x_ = D[x]['y'], D[x]['s'], D[x]['k'], D[x]['x_']
        bits, B = B[:k][::-1], B[k:]
        print(f"\nbits emitted: {bits}, leftover stream: {B}")
        x = x_ + int(bits, 2)
        print(f"updated x: {x}")
        stream = s + stream        
    
    return stream
        

stream = '1121211020111021'
x = 19
B, x_f = Encoder(stream, x)
print(f"\nthe bitsteam is: {B}\nthe final state is: {x_f}")
stream_decoded = Decoder(B, x_f)
print(f"\nthe decoded stream: {stream_decoded}")
assert(stream == stream_decoded)