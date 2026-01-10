from datasets import load_dataset
from tqdm import tqdm

def count_rows():
    print("Loading dataset G-reen/instruct-set-longer-fixed in streaming mode...")
    ds = load_dataset("G-reen/instruct-set-longer-fixed", split="train", streaming=True)
    
    count = 0
    print("Counting rows...")
    for _ in tqdm(ds):
        count += 1
        
    print(f"Total rows: {count}")

if __name__ == "__main__":
    count_rows()
