import numpy as np
from tqdm import tqdm
from network import XORNet

def main():
    # 1. Data (XOR)
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    # 2. Initialize
    print("Initializing NumPyNet...")
    nn = XORNet(input_size=2, hidden_size=4, output_size=1)

    # 3. Train
    epochs = 10000

    lr = 0.1
    
    print(f"Training for {epochs} epochs...")
    pbar = tqdm(range(epochs))
    
    for i in pbar:
        nn.forward(X)
        nn.backward(X, y, lr)
        
        if i % 100 == 0:
            loss = np.mean(np.square(y - nn.y_hat))
            pbar.set_description(f"Loss: {loss:.4f}")

    # 4. Test
    print("\nFinal Predictions:")
    preds = nn.forward(X)
    print(preds)
    
    # Simple check
    if preds[0] < 0.1 and preds[1] > 0.9:
        print("\n✅ Success: The Network learned XOR logic!")
    else:
        print("\n❌ Failure: The Network did not converge.")

if __name__ == "__main__":
    main()