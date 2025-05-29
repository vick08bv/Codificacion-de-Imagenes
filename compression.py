import heapq
import numpy as np
from collections import defaultdict


# Codificación RLE de una imagen en blanco y negro
def rle_encode(image):
    rle = []
    # Imagen vectorizada
    vector = image.flatten()

    run = 1
    current_value = vector[0]

    for i in range(1, len(vector)):
        # Se verifica si se amplia el run o empieza otro.
        if vector[i] == current_value:
            run += 1
        else:
            rle.append((current_value, run))
            current_value = vector[i]
            run = 1

    # Runs representados por valor y longitud
    rle.append((current_value, run))
    return rle


def entropy_rle(lengths):
    values, counts = np.unique(lengths, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))


# Bits por pixel
def bpp_rle(encoded_image, image_bytes):
    # Tamaño comprimido, por cada tupla se tiene 1 byte por simbolo y 4 bytes por longitud
    return (len(encoded_image) * (1 + 4) * 8) / image_bytes


# Clase para crear el árbol de Huffman
class Node:
    def __init__(self, value, freq):
        self.symbol = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


# Creación del árbol de Huffman
def build_huffman_tree(frequencies):
    heap = [Node(sym, freq) for sym, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(None, n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heapq.heappush(heap, merged)
    return heap[0]


# Mapeo bajo el algoritmo de Huffman
def generate_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node is not None:
        if node.symbol is not None:
            codes[node.symbol] = prefix
        # Asignación de cadenas binarias
        generate_codes(node.left, prefix + "0", codes)
        generate_codes(node.right, prefix + "1", codes)
    return codes


# Codificación de Huffman
def huffman_encode(flat_image):
    # Frecuencia de cada símbolo
    freqs = defaultdict(int)
    for val in flat_image:
        freqs[val] += 1
    # Árbol y mapeo
    tree = build_huffman_tree(freqs)
    codes = generate_codes(tree)
    encoded_bits = ''.join(codes[val] for val in flat_image)
    return encoded_bits, freqs, codes


# Entropía alcanzada por la codificación de Huffman
def entropy_huffman(frequencies):
    total = sum(frequencies.values())
    entropy = -sum((count / total) * np.log2(count / total) for count in frequencies.values())
    return entropy


# Bits por pixel
def bpp_huffman(encoded_bits, image_flattened):
    # Bits codificados sobre el total de pixeles originales
    return len(encoded_bits) / len(image_flattened)
