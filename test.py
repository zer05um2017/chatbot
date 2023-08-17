import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # with open('medi/rtdocs\www.kahp.or.kr/socialcontribution.do.html', encoding='utf-8') as f:
    #     print(f.read())
    
    # keys = {}
    load_dotenv()
    print(os.environ.get("OPENAI_API_KEY"))
    
    # with open("./.env", "r") as f:
    #     key_list = f.readlines()
    #     print(type(key_list))
    #     key_list.
        
            