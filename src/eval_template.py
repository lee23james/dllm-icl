import random
from typing import List, Tuple
from transformers import AutoTokenizer

#这里是专门为了进行eval测试进行的,所以还是需要question中的answer,而不需要mask
#目前先考虑sudoku和countdown
SUDOKU_SYSTEM_PROMPT ="""Solve this 4x4 Sudoku puzzle represented as a 16-digit string (read left-to-right, top-to-bottom) where '0'=empty cell.

Requirements:
1. Replace ALL '0's with digits 1-4
2. Follow STRICT Sudoku rules:
   - Rows: Each must contain 1-4 exactly once
   - Columns: Each must contain 1-4 exactly once
   - 2x2 Boxes: Each must contain 1-4 exactly once
3. Format answer as:
<answer>
[16-digit solution]
</answer>"""


SUDOKU_PROMPT = """Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells."""



# SUDOKU_shot1="""Puzzle: 
# 4100
# 0001
# 1300
# 2000
# <answer> 
# 4132
# 3241
# 1324
# 2413
# </answer>"""

SUDOKU_shot1="""Puzzle: 
3102
2000
0210
0320
<answer> 
3142
2431
4213
1324
</answer>"""

# SUDOKU_shot1="""Puzzle: 
# 2100
# 0310
# 0421
# 1000
# <answer> 
# 2143
# 4312
# 3421
# 1234
# </answer>"""




# SUDOKU_shot2="""Puzzle: 
# 0004
# 0321
# 0203
# 3002
# <answer>
# 2134
# 4321
# 1243
# 3412
# </answer>"""

SUDOKU_shot2="""Puzzle: 
4310
0100
0030
3420
<answer>
4312
2143
1234
3421
</answer>"""

# SUDOKU_shot2="""Puzzle: 
# 4020
# 2314
# 0200
# 1000
# <answer>
# 4123
# 2314
# 3241
# 1432
# </answer>"""






# SUDOKU_shot3="""Puzzle: 
# 4123
# 0000
# 0402
# 2300
# <answer>
# 4123
# 3214
# 1432
# 2341
# </answer>"""

SUDOKU_shot3="""Puzzle: 
2004
0012
0320
1003
<answer>
2134
3412
4321
1243
</answer>"""

# SUDOKU_shot3="""Puzzle: 
# 0031
# 3000
# 0042
# 2403
# <answer>
# 4231
# 3124
# 1342
# 2413
# </answer>"""






# SUDOKU_shot4="""Puzzle:
# 1432
# 0041
# 3000
# 4000
# <answer>
# 1432
# 2341
# 3214
# 4123
# </answer>"""

SUDOKU_shot4="""Puzzle:
0024
0031
0100
0413
<answer>
1324
4231
3142
2413
</answer>"""

# SUDOKU_shot4="""Puzzle:
# 0000
# 2413
# 0231
# 0100
# <answer>
# 1324
# 2413
# 4231
# 3142
# </answer>"""






# SUDOKU_shot5="""Puzzle:
# 0020
# 0341
# 0210
# 1002
# <answer>
# 4123
# 2341
# 3214
# 1432
# </answer>"""

SUDOKU_shot5="""Puzzle:
2013
1320
0040
0200
<answer>
2413
1324
3142
4231
</answer>"""

# SUDOKU_shot5="""Puzzle:
# 0104
# 3401
# 4003
# 1000
# <answer>
# 2134
# 3421
# 4213
# 1342
# </answer>"""


#需要的是answer_str同时进行评估和测评
def sudoku_prompt_eval(puzzle_str: str,answer_str: str,query_position): # prompt for sudoku
    #构造不同位置的prompt,这里是通过question_shot中的question和answer来进行构造,最后要据此分成两个部分,一个是左边的prompt,一个是右边的prompt还有一个是答案prompt
    puzzle_str = '\n'.join(puzzle_str[i:i+4] for i in range(0, len(puzzle_str), 4))
    question_shot=f"""Puzzle:
{puzzle_str}
<answer>
"""
    #这里可以依次构建shot来进行拆分
    answer_shot='\n'.join(answer_str[i:i+4] for i in range(0, len(answer_str), 4))
    # question_shot=puzzle_str

        #这里需要去测试一下,其他shot我目前想测的是其4shot的情况,看看我想的是不是这样,这些规律能不能泛化到其他数据集
    if query_position == 0:
        combined_prompt =  SUDOKU_PROMPT + "\n\n" + SUDOKU_shot1 + "\n\n" + SUDOKU_shot2 + "\n\n" + SUDOKU_shot3 + "\n\n" + SUDOKU_shot4 + "\n\n" + SUDOKU_shot5 + "\n\n" + question_shot 
        right_prompt = "\n</answer>"
    elif query_position == 1:
        combined_prompt = SUDOKU_PROMPT + "\n\n" + SUDOKU_shot1 + "\n\n" + SUDOKU_shot2 + "\n\n" + SUDOKU_shot3 + "\n\n" + SUDOKU_shot4 + "\n\n" + question_shot
        right_prompt="\n</answer>"+"\n\n" + SUDOKU_shot5
    elif query_position == 2:
        combined_prompt = SUDOKU_PROMPT +  "\n\n" + SUDOKU_shot1 + "\n\n" + SUDOKU_shot2 + "\n\n" + SUDOKU_shot3 + "\n\n" + question_shot
        right_prompt="\n</answer>"+"\n\n" + SUDOKU_shot4 + "\n\n" + SUDOKU_shot5
    elif query_position == 3:
        combined_prompt = SUDOKU_PROMPT + "\n\n" + SUDOKU_shot1 + "\n\n" + SUDOKU_shot2 + "\n\n" + question_shot 
        right_prompt="\n</answer>"+"\n\n" + SUDOKU_shot3 + "\n\n" + SUDOKU_shot4 + "\n\n" + SUDOKU_shot5
    elif query_position == 4:
        combined_prompt = SUDOKU_PROMPT + "\n\n" + SUDOKU_shot1 + "\n\n" + question_shot 
        right_prompt="\n</answer>"+"\n\n" + SUDOKU_shot2 + "\n\n" + SUDOKU_shot3 + "\n\n" + SUDOKU_shot4 + "\n\n" + SUDOKU_shot5
    elif query_position == 5:
        combined_prompt = SUDOKU_PROMPT + "\n\n" + question_shot 
        right_prompt="\n</answer>"+"\n\n" + SUDOKU_shot1 + "\n\n" + SUDOKU_shot2 + "\n\n" + SUDOKU_shot3 + "\n\n" + SUDOKU_shot4 + "\n\n" + SUDOKU_shot5
    
    #这个是4shot的情况
    elif query_position == 6:
        combined_prompt = SUDOKU_PROMPT+  "\n\n"+SUDOKU_shot1+ "\n\n"+SUDOKU_shot2 + "\n\n" + SUDOKU_shot3+ "\n\n" + SUDOKU_shot4+ "\n\n"+question_shot 
        right_prompt= "\n</answer>"
    elif query_position == 7:
        combined_prompt = SUDOKU_PROMPT+  "\n\n"+SUDOKU_shot1+ "\n\n"+SUDOKU_shot2+ "\n\n" + SUDOKU_shot3+ "\n\n"+question_shot 
        right_prompt="\n</answer>"+"\n\n" + SUDOKU_shot4
    elif query_position == 8:
        combined_prompt = SUDOKU_PROMPT+  "\n\n"+SUDOKU_shot1+ "\n\n" +SUDOKU_shot2+ "\n\n"+question_shot 
        right_prompt="\n</answer>"+"\n\n" + SUDOKU_shot3+ "\n\n" + SUDOKU_shot4
    elif query_position == 9:
        combined_prompt = SUDOKU_PROMPT+  "\n\n"+SUDOKU_shot1+ "\n\n"+question_shot 
        right_prompt="\n</answer>"+"\n\n" +SUDOKU_shot2+ "\n\n" + SUDOKU_shot3+ "\n\n" + SUDOKU_shot4
    elif query_position == 10:
        combined_prompt = SUDOKU_PROMPT+ "\n\n"+question_shot 
        right_prompt= "\n</answer>"+"\n\n"+SUDOKU_shot1+ "\n\n" +SUDOKU_shot2+ "\n\n" + SUDOKU_shot3+ "\n\n" + SUDOKU_shot4
        
        
        
    else:
        raise ValueError(f"Invalid query position: {query_position}")


    #返回左边中间和右边,方便进行评估
    return SUDOKU_SYSTEM_PROMPT + '\n\n' + combined_prompt,answer_shot,right_prompt


#针对countdown数据集进行的实验
def countdown_prompt_eval(question,query_position,answer_str): # prompt for countdown
    front_shot = '''For the given numbers, find a sequence of arithmetic operations that results in the last number.
Place the sequence of operations after Solution: tags.'''

    shot1='''Question: 
15,44,79,50
Solution:
<answer>
44-15=29,79-29=50
</answer>'''

    shot2='''Question: 
1,2,12,25
Solution:
<answer> 
2*12=24,1+24=25
</answer>'''

    shot3='''Question: 
3,85,5,30
Solution: 
<answer>
85+5=90,90/3=30
</answer>'''

    question_shot=f'''Question: 
{question}
Solution: 
<answer>
'''

    answer_shot=answer_str
    
    if query_position == 0:
        combined_prompt = front_shot+"\n\n"+shot1 + "\n\n" + shot2 + "\n\n" + shot3 + "\n\n" + question_shot
        right_prompt = "\n</answer>"
    elif query_position == 1:
        combined_prompt = front_shot+"\n\n"+shot1 + "\n\n" + shot2 + "\n\n" +question_shot
        right_prompt="\n</answer>"+"\n\n" + shot3
    elif query_position == 2:
        combined_prompt = front_shot+"\n\n"+shot1+ "\n\n" +question_shot
        right_prompt="\n</answer>"+"\n\n" + shot2 + "\n\n"   +shot3
    elif query_position == 3:
        combined_prompt = front_shot+ "\n\n" +question_shot
        right_prompt="\n</answer>"+"\n\n"+shot1 + "\n\n"+  shot2 + "\n\n"  + shot3 
    else:
        raise ValueError(f"Invalid query position: {query_position}")
    return combined_prompt,answer_shot,right_prompt


def main():
    print("--------------------------------测试sudoku指令:构造puzzle看看不同位置的情况:--------------------------------")
    sudoku_puzzle="4100000104004032"
    sudoku_answer="4132324113242413"
    for position in range(6):
        sudoku_prompt_result=sudoku_prompt_eval(sudoku_puzzle,sudoku_answer,position)
        print(f"position: \n{position}\nprompt: \n{sudoku_prompt_result}")
        print(sudoku_prompt_result[0]+sudoku_prompt_result[1]+sudoku_prompt_result[2])

    print("--------------------------------测试countdown指令:构造question看看不同位置的情况:--------------------------------")
    countdown_question="15,44,79,50"
    countdown_answer="44-15=29,79-29=50"
    for position in range(4):
        countdown_prompt_result=countdown_prompt_eval(countdown_question,position,countdown_answer)
        print(f"position: \n{position}\nprompt: \n{countdown_prompt_result}")
        print(countdown_prompt_result[0]+countdown_prompt_result[1]+countdown_prompt_result[2])
if __name__ == "__main__":
    main()