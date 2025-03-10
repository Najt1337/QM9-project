#%%
import re
#%%
class SMILESTokenizer:
    def __init__(self):
        """
        Initializes the SMILES tokenizer with a vocabulary and token-to-ID mapping.
        """
        # Vocabulary list
        all_atoms = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        ]

        other_smiles_characters = [
            '-', '=', '#', ':', '(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '@', '/', '\\', '[', ']', '.', '%', '*', 'b', 'c', 'n', 'o', 'p', 's', '&', '^', '~'
        ]

        # Combine all atoms and other SMILES characters to form the vocabulary
        self.vocabulary = all_atoms + other_smiles_characters

        # Add special tokens (optional, e.g., for padding, unknown tokens, etc.)
        self.special_tokens = ['<PAD>', '<UNK>']
        self.vocabulary = self.special_tokens + self.vocabulary

        # Create a token-to-ID mapping
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}

        # Create an ID-to-token mapping (optional, for decoding)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        # Regular expression pattern to match SMILES tokens
        self.pattern = r"(\[[^]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|:|\\|\/|@|\+|~|\d+|\%\d\d|\[|\])"

    def tokenize(self, smiles):
        """
        Tokenizes a SMILES string into a list of tokens.
        
        Args:
            smiles (str): The SMILES string to tokenize.
        
        Returns:
            list: A list of tokens.
        """
        return re.findall(self.pattern, smiles)

    def encode(self, smiles):
        """
        Converts a SMILES string into a list of token IDs.
        
        Args:
            smiles (str): The SMILES string to encode.
        
        Returns:
            list: A list of token IDs.
        """
        tokens = self.tokenize(smiles)
        return [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]

    def decode(self, token_ids):
        """
        Converts a list of token IDs back into a SMILES string.
        
        Args:
            token_ids (list): A list of token IDs.
        
        Returns:
            str: The decoded SMILES string.
        """
        tokens = [self.id_to_token.get(token_id, '<UNK>') for token_id in token_ids]
        return ''.join(tokens)
#%%
# Example usage
tokenizer = SMILESTokenizer()

# SMILES string
smiles_string = "CCO[N+](=O)[O-]"

# Step 1: Tokenize the SMILES string
tokens = tokenizer.tokenize(smiles_string)
print("Tokens:", tokens)

# Step 2: Encode the tokens into IDs
token_ids = tokenizer.encode(smiles_string)
print("Token IDs:", token_ids)

# Step 3: Decode the token IDs back into a SMILES string
decoded_smiles = tokenizer.decode(token_ids)
print("Decoded SMILES:", decoded_smiles)