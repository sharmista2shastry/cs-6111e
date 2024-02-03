'''
COMS E6111 - Project 1
Sharmista Shastry (ss6950) and Cindy Ruan (cxr2000)
'''

import sys

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) != 5:
        print("Usage: python main.py <google api key> <google engine id> <precision> <query>")
        sys.exit(1)

    # Get command line arguments
    api_key = sys.argv[1]
    engine_id = sys.argv[2]
    precision = sys.argv[3]
    query = sys.argv[4]

    # Check if precision is between 0 and 1
    try:
        precision = float(precision)
        if precision < 0 or precision > 1:
            raise ValueError
    except ValueError:
        print("Precision must be between 0 and 1")
        sys.exit(1)

    
