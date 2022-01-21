from math import log, floor, ceil
import random

def adaptive_input_processing():
    
    stream = input("\nEnter your symbol stream:")
    alphabet = set(list(stream))
    f_s = {s : 0 for s in alphabet}
    for s in list(stream):
        f_s[s] += 1
  
    return stream, f_s


def choose_distribution(type=1):

    d = {1: {'0': 3, '1':8, '2': 5}, 2: {'a':5, 'b':6, 'c':7, 'd':14}, 3: {'u':4, 'p':7, 'y':5}}
    distribution = {}
    if type == 1:
        # distribution and alphabet are set
        choice = int(input(f"\nEnter corresponding number to pick preset alphabet and distribution\n [1] {d[1]}\n [2] {d[2]}\n [3] {d[3]}\nEnter here: "))
        distribution = d[choice]
    elif type == 2:
        # user enters a dictionary of k : v :: alphabet : fq
        user = input("\nEnter key:value pairs separated by a comma and space.\nFor example: a:10, b:2, c:4\n")
        for thing in user.split(", "):
            pair = thing.split(":")
            k, v = pair[0], int(pair[1])
            distribution.update({k:v})
    else:
        print("\nInvalid choice of distribution type.")

    print(f"\nThe distribution you chose is: {distribution}")
    return distribution


def table_primitives(f_s):
    
    cumul = sum(f_s.values())
    R = int(ceil(log(cumul, 2)))
    L = 1 << R
    q_s = {s: L*f//cumul for s, f in f_s.items()}
    state_labels = [l for l in range(L, 2*L)]
    state_labels_s = {s : [l for l in range(f, 2*f)] for s, f in q_s.items()}
    all_pre_labels = []
    for symbol, l in state_labels_s.items():
        all_pre_labels += [(i, symbol) for i in l]
    
    return R, L, q_s, state_labels, state_labels_s, all_pre_labels

def symbol_spread(L, R, all_pre_labels):
    spread = [0] * L
    print(f"\nlen of spread : {len(spread)}\nlen of all_pre_labels = {len(all_pre_labels)}")
    for i in range(L):
        index = (i + R * (1 % R)) % L # change from R to 4 ?
        print(f"\nindex calculated for i = {i} is : {index} .") # and all_pre_labels[i] = {all_pre_labels[i]}
        spread[index] = all_pre_labels[i] # creates list index out of range problem for adaptive and there aren't as many pre-labels as there are labels.

    return spread

def scramble(L, R, spread):
    # currently only for L = 16
    if L == 16:
        scrambled_spread = [None] * L
        key = '2222111133330000'
        for i in range(L):
            block = i//R
            inblock_shift = int(key[i])
            inblock_index = (i - inblock_shift) % 4
            index = 4 * block + inblock_index
            scrambled_spread[i] = spread[index]
        
        return scrambled_spread
    else:
        print(f"Scramble currently not supported for L = {L}")

    return spread


def build_decoding_table(q_s, state_labels, spread, state_labels_s, L, R):
    D = {k: None for k in state_labels}
    ctr = {s: 0 for s in q_s.keys()}

    for state_label in D.keys():
        D[state_label] = {}
        s = spread[state_label - L][1] # the corresponding symbol
        symbol_list = state_labels_s[s]

        D[state_label]['y'] = symbol_list[ctr[s]]
        D[state_label]['s'] = s
        D[state_label]['k'] = R - int(log(D[state_label]['y'], 2)) # num bits to read out of stream
        D[state_label]['x_'] = D[state_label]['y'] << D[state_label]['k'] # next state to go to is x_ + decimal(k extracted msbits)
        ctr[s] += 1

    return D

def build_encoding_table(q_s, state_labels, state_labels_s, C):
    E = {k : {s : None for s in q_s.keys()} for k in state_labels}
    for x in E.keys():
        for s in q_s.keys():
            L_s = len(state_labels_s[s])
            k = int(log((x/L_s), 2))
            y = floor(x/(1 << k))
            bits = bin(int(x % (1 << k)))[2 : ].zfill(k)
            E[x][s] = (C[y][s], bits) # this is where the encoding function is useful
    
    return E

def Encoder(stream, x, E):
    n = len(stream)
    B = ""
    x = x
    for i in range(n):
        s = stream[i]
        x, bits = E[x][s]
        B += bits
    
    return B, x


def Decoder(B, x_f, D):
    x = x_f
    B = B[ : : -1]
    stream = ""
    while len(B) > 0:
        y, s, k, x_ = D[x]['y'], D[x]['s'], D[x]['k'], D[x]['x_']
        if k == 0:
            # what to do? enters inft loop when k=0. mitigate this and you are done
            continue

        bits, B = B[ : k][ : : -1], B[k : ]
        # to debug
        print(f"x (before) : {x}\nx_ (during) : {x_}\nbits : {bits}, converted to int : {int(bits, 2)}\nbitstream : {B}")
        x = x_ + int(bits, 2)
        print(f"x (after) : {x}\n")
        stream = s + stream
    
    return stream


def main():

    ############################
    ##     INITIALIZATION     ##
    ############################

    choices = {1: 'Adaptive Distribution', 2:  'Preset distribution and flexible input string'}
    choice = input(f"\n\n-----------------------------------------------\nTabled Variant of Asymmetrical Numeral Systems\n-----------------------------------------------\n\nEnter the corresponding number to pick your choice of input:\n [1] {choices[1]}\n [2] {choices[2]}\n\nEnter here: ")
    print(f"Your choice was {choice}: {choices[int(choice)]}")

    stream, f_s, R, L, q_s, state_labels, state_labels_s, all_pre_labels = None, None, None, None, None, None, None, None
    if choice == '1':
        stream, f_s = adaptive_input_processing()
        
    elif choice == '2':
        c = input("\nDo you want to enter the distribution? Enter 'y' if yes, any other key if no.\nEnter here: ")        
        if c == 'y':
            f_s = choose_distribution(2)
        else:
            f_s = choose_distribution()

        while True:
            stream = input("\nEnter the stream you want to compress: ")
            alphabet, alphabet_  = set([k for k in f_s.keys()]), set(list(stream))
            if alphabet == alphabet_:
                f_s_in_stream = {s : 0 for s in alphabet}
                for s in list(stream):
                    f_s_in_stream[s] += 1
                
                for k in alphabet:
                    if f_s[k] != f_s_in_stream[k]:
                        print("\nThere is a difference in frequencies of alphabets in the message and in the distribution.\nThe compression will be less than optimal.")
                        break
                break
            else:
                print(f"\nThe stream you entered does not have the same alphabet as the one you chose. The following symbols are disallowed : {alphabet_ - alphabet}.")
                continue
    
    # constants
    R, L, q_s, state_labels, state_labels_s, all_pre_labels = table_primitives(f_s)      
    print(f"\ninput stream : {stream}\nf_s : {f_s}\nR : {R}\nL : {L}\nq_s : {q_s}\nstate_labels : {state_labels}\nstate_labels_s : {state_labels_s}\nall_pre_labels : {all_pre_labels}\n")

    after_spread = symbol_spread(L, R, all_pre_labels)

    if input("\nEnter 's' to scramble, anything else otherwise: ") == 's':
        after_spread = scramble(L, R, after_spread)

    # decoding table and function
    D = build_decoding_table(q_s, state_labels, after_spread, state_labels_s, L, R)
    print(f"\nDecoding table: \n{D}")

    # enumerating the encoding function C(s, y) = x
    l = set([k[0] for k in all_pre_labels])
    inv_map = {(d['s'], d['y']) : k for k, d in D.items()}
    C = {k : {s : None for s in q_s.keys()} for k in l}

    for k in C.keys():
        for t in inv_map.keys():
            if t[1] == k:
                s = t[0]
                C[k][s] = inv_map[t]
    
    # encoding table
    E = build_encoding_table(q_s, state_labels, state_labels_s, C)
    print(f"\nEncoding table: \n{E}")

    print("\n\nNow, onto encoding and decoding the stream you gave as input and testing if the process is correct.\n")

    ######################
    ##     ENCODING     ##
    ######################
    x = state_labels[random.choice([1, 9, 4])] # make this a random number in the lower range of the labels
    print("The starting state: ", x)
    B, x_f = Encoder(stream, x, E)
    print(f"\nThe bitstream is : {B}\nThe final state is : {x_f}\nThe length of the bitstream: {len(B)}")


    ######################
    ##     DECODING     ##
    ######################
    stream_decoded = Decoder(B, x_f, D)
    print(f"\nthe decoded stream: {stream_decoded}")
    print(f"The stream encoded: {stream}")
    assert(stream == stream_decoded)


main()