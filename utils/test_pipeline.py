import sys
from transformers import pipeline

fill_masker = pipeline("fill-mask", model=sys.argv[1])
fill_masker("London is the capital of[MASK].")
