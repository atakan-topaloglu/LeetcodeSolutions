import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Optional, TypeVar
from collections import Counter, deque
from pprint import pprint

# Longest Palindromic Substring

def longest_palindromic_sub(s:str)->set:
    """Detects the longest palindromic substring from a given string.
    If there are distinct palindromic substrings of same length, returns both.
    A palindrome is a sequence of symbols that read the same backwards as forwards."""

    longest_subs = []
    longest_sub = []

    for i in range(len(s)):
        j = 1
        sub = [s[i]]
        while j < min(len(s)-i, i+1) and s[i-j] == s[i+j]:
            sub.insert(0, s[i-j])
            sub.append(s[i+j])
            j += 1

        if len(sub)>len(longest_sub):
            longest_sub = sub
            longest_subs = []
            longest_subs.append(sub)
        elif len(sub) == len(longest_sub):
            longest_subs.append(sub)

    return set(["".join(palin_sub) for palin_sub in longest_subs])

# Palindromic number

def is_palindrome_number(num: int)->bool:
    num = str(num)
    return num == num[::-1]


# Valid Paranthesis

def valid_paranthesis(s: str)->bool:
    """Computes if the a s of three different types of opening and closing parantheses is valid.
    Opening and closing paranthesis pairs are: (), [], {}.
    """
    
    chars = {")": "(", "}": "{", "]": "["}
    char_stack = []
    for i in s:
        if i in chars.values():
            char_stack.append(i)
        elif i in chars.keys():
            if char_stack == [] or char_stack[-1] != chars[i]:
                return False
            char_stack.pop()
    return char_stack == []


# Longest Substring without repeating characters

def lengthOfLongestSubstring(s:str)-> int:
    ptr0 = 0
    ptr1 = 1
    longest_substring = []
    substring = []
    while ptr1 < len(s):
        if s[ptr1] not in substring:
            substring.append(s[ptr1]) 
            ptr1 += 1
        else:
            pass
    return substring


# Two Sum 

def twoSum(nums:List[int], target:int) -> List[int]:
    for num1 in range(len(nums)):
        for num2 in range(len(nums)):
            if nums[num1] + nums[num2] == target and num1 != num2:
                return [num1, num2]

# Efficient Two Sum

def twoSumEfficient(nums:list, target:int) -> list:
    map = {}
    for num_idx, num in enumerate(nums):
        if num not in map.keys():
            map[target - num] = num_idx
        else:
            return num_idx, map[num]



def uniqueOccurences(arr:list)->bool:
    """Calcualates the total number of occurences for each element in array.
    Returns if there are two elements with same number of occurences."""
    
    word_freq = Counter(arr)
    frequencies = list(word_freq.values())
    all_freqs_unique = len(set(frequencies)) == len(frequencies)
    return all_freqs_unique


# Merge two ordered linked lists into a new linked list.

class Node:

    "Initializes the nodes in a linked list"

    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedList:

    def __init__(self):
        self.head = None


    def append(self, val):
        "Appends the value to the Linked List"

        # Special case if we're appending to an empty linked list.
        new_node = Node(val)
        if self.head is None:
            self.head = new_node
            return
        
        temp = self.head
        while temp.next is not None: #rather than doing is not None, you can leave it as "while temp.next"
            temp = temp.next
        temp.next = new_node
    

    def get_node(self, index:int):
        "Helper function for insert to get the index of the previous node for new node to be inserted"
        
        if index<0:
            raise IndexError("Index value has to be greater than 0")
        temp = self.head

        while index>0:
            temp = temp.next
            if temp is None:
                raise Exception("There are fewer nodes in the linked list than the index specified.")
            index -= 1
        return temp


    def insert(self, index:int, val):
        "Inserts a value to the specified index of the Linked List"

        new_node = Node(val)

        # If we're inserting into an empty Linked List, 
        # the self.get_node() below will raise a negative index exception
        # We write a special case for this 
        if index == 0:
            new_node.next = self.head
            self.head = new_node
            return
        prev_node = self.get_node(index - 1)

        new_node.next = prev_node.next
        prev_node.next = new_node
            

    def get_prev_node2(self, val):
        "Helper function for delete, returns the parent node of the node to be deleted"

        temp = self.head
        # Check how to handle if we try to get the value of the previous node in a length 1 linked list.
        # Return None doesn't work bc we try to access node.value, raises an NoneType has no attr 'val' error. 
        # Or leave it as is, and check if null value in the post process

        while temp is not None:
            if temp.next is None:
                raise Exception("Node not in the linked list")
            elif temp.next.val == val:
                return temp
            temp = temp.next
        

    def get_prev_node(self, val):
        "Helper function for delete, returns the parent node of the node to be deleted"

        temp = self.head
        while temp.next is not None:
            if temp.next.val == val:
                return temp
            temp = temp.next


    def delete(self, val):
        "Deletes the element with the given value form the linked list."

        # Edge case if the node to be deleted is the first node.
        temp = self.head
        if temp.val == val:
            if temp.next is None:
                self.head = None
                return 
            temp.val = temp.next.val
            temp.next = temp.next.next
            return

        prev_node = self.get_prev_node(val)
        del_node = prev_node.next
        next_node = del_node.next
        prev_node.next = next_node


    # Implement delete functionality using the index, similar to how insert worked.

    def printLL(self):
        "Prints the nodes of the Linked List"

        temp = self.head
        if temp.next is None or temp is None: # Also check this condition. Do we really need to check both conditions, do we need more?
            print("The linked list is empty")
        while temp is not None: 
            print(temp.val, "->" ,end=" ")
            temp = temp.next



# Is this solution the same as the solution below (which is valid)
class Solution2:
    def mergeTwoLists2(self, list1:Optional[Node], list2:Optional[Node])->Optional[Node]:
        temp = Node
        while list1 is not None and list2 is not None:
            if list1.val > list2.val:
                temp.next = list1
                list1 = list1.next #Didnt really understand this. 
            else:
                temp.next = list2
                list2 = list1.next
            temp = temp.next # And didn't really understand this, check these.
        if list1 is not None and list2 is None:
            temp = list1
        else:
            temp = list2


class Solution:
    def mergeTwoLists(self, list1:Optional[Node], list2:Optional[Node])->Optional[Node]:
        dummy = Node
        tail = dummy
        
        while list1 and list2:
            if list1.val > list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next

        tail.next = list1 or list2

        return dummy.next
# Middle of a linked list

# Jump Game solved using Dynamic Programming

# First Attempt
def is_valid_jumpgame(arr)->bool:
    pos_reachable = [False for i in arr]
    pos_reachable[0] = True

    for i in range(len(arr)):
        if pos_reachable[i]:
            for j in range(min(arr[i]+1, len(arr)-i)):
                pos_reachable[i + j] = True

    return pos_reachable[-1]


# Determine if two strings are close

def closeStrings(word1:str, word2:str)->bool:
    """Returns if two strings are close.
    Code below explains the requirements of close strings better than any docstring can."""

    count_word1 = Counter(word1)
    count_word2 = Counter(word2)

    have_same_letters = set(count_word1.keys()) == set(count_word2.keys())
    have_same_freqs = sorted(count_word1.values()) == sorted(count_word2.values())

    return have_same_letters and have_same_freqs


# Sort Characters by Frequency

def frequencySort(s:str)->str:
    k = Counter(s)
    k = sorted(k.items(), key=lambda pair: pair[1], reverse=True)
    output_str = ''.join([i[0]*i[1] for i in k])
    return output_str


# Minimum Average Difference

def minimumAverageDifference(nums: List[int]) -> int:
    """For each number (pivot) in the list, calculates the average of numbers to the left and right of the selected number.
    Then takes the absolute difference between these two averages for each pivot.
    Returns the index of the pivot number that has the lowest absolute difference between averages."""

    avg_diff_list = []
    sum_first = 0
    sum_last = sum(nums)

    for i in range(len(nums)-1):
        sum_first += nums[i]
        sum_last -= nums[i]

        avg_first = int(sum_first/(i+1))
        avg_last = int(sum_last/(len(nums)-1-i))

        avg_diff = abs(avg_first - avg_last), i
        avg_diff_list.append(avg_diff)

    # Calculating the average difference for the last index
    sum_first += nums[-1] 
    # It's not necessary to calculate sum_last since sum_last will be 0 for the last element.
    
    avg_diff_final = int(sum_first/len(nums)), len(nums)-1
    avg_diff_list.append(avg_diff_final)

    minimum_avg_dist_idx = min(avg_diff_list)[1]

    return minimum_avg_dist_idx

# Climbing Stairs

def climbingStairs(n:int)->int:
    "Returns the number of unique ways stairs can be climbed using step sizes 1,2"
    stair_climbs = [0 for i in range(n+1)]
    stair_climbs[1] = 1 # 1
    stair_climbs[2] = 2 # 1,1 or 2

    for i in range(3, n+1):
        stair_climbs[i] = stair_climbs[i-1] + stair_climbs[i-2]
    final_ways = stair_climbs[n]
    return final_ways

# We can acutally generalize this
# Use a bottom up approach, imagine a staircase with n (6) stairs
# If you're on 6 can reach 6th stair in 0 ways [you can only take 1 or 2 step sized steps]
# From 5- 1
# From 4- 1,1 or 2
# From 3- only consider reaching 5 or 4, if you CONNECT with 5 or 4, you can add the ways from 5 and 4 and store it in 3.
# Do the same for 2,1,0. [only consider n-1 and n-2 steps and add them up]

# Furthermore if you have steps sized 1,2,4, again with n (6) steps.
# From 5- 1; From 4- 1,1 or 2; From 3- 1,1,1 or 1,2 or 2,1; From 2- 4 or 2,2 or 2,1,1 or 1,2,1 or 1,1,2 or 1,1,1,1
# Now, if you CONNECT with n-1, n-2 and n-4 th steps, you simply add up their unique possibilities.


def climbingStairs_2(n:int)->int:
    "Returns the number of unique ways stairs can be climbed using step sizes 1,2 and 4."

    stair_climbs = [0 for i in range(n+1)]
    stair_climbs[1] = 1 
    stair_climbs[2] = 2 
    stair_climbs[3] = 3
    stair_climbs[4] = 6

    for i in range(5, n+1):
        stair_climbs[i] = stair_climbs[i-1] + stair_climbs[i-2] + stair_climbs[i-4]
    final_ways = stair_climbs[n]
    print(stair_climbs)
    return final_ways

# Most optimized version of climbingstairs_2_1.
def climbingStairs_2_1(n:int)->int:
    """Optimized version of climbingStairs_2. 
    Returns the number of unique ways stairs can be climbed using step sizes 1,2 and 4."""

    # We need to hardcode the initial conditions.
    stair_climbs_minus4 = 1
    stair_climbs_minus3 = 2
    stair_climbs_minus2 = 3
    stair_climbs_minus1 = 6

    stair_climbs_cur = stair_climbs_minus1 + stair_climbs_minus2 + stair_climbs_minus4
    
    for i in range(n-5):
        stair_climbs_minus4 = stair_climbs_minus3
        stair_climbs_minus3 = stair_climbs_minus2
        stair_climbs_minus2 = stair_climbs_minus1
        stair_climbs_minus1 = stair_climbs_cur

        stair_climbs_cur = stair_climbs_minus1 + stair_climbs_minus2 + stair_climbs_minus4

    return stair_climbs_cur

# Halves Are Alike
def halvesAreAlike(s:str):
    s = s.lower()
    str_left = s[:(len(s)//2)]
    str_right = s[len(s)//2:]

    vowels = ["a", "e", "i", "o", "u"]
    str_left_vowels = len([letter for letter in str_left if letter in vowels])
    str_right_vowels = len([letter for letter in str_right if letter in vowels])
    return str_left_vowels == str_right_vowels


# How to get the islands in a graph

def neighboring_vertices(i:int, j:int, matrix):
    #Returns the values of the neighboring cells
    matrix_y_dim, matrix_x_dim = matrix.shape
    if i==0 and j==0:
        return (matrix[i,j+1], matrix[i+1,j])
    if i==0 and j==matrix_x_dim-1:
        return (matrix[i,j-1], matrix[i+1,j])
    if i==matrix_y_dim-1 and j==0:
        return (matrix[i,j+1], matrix[i-1,j])
    if i==matrix_y_dim-1 and j==matrix_x_dim-1:
        return (matrix[i,j-1], matrix[i-1,j])
    if i==0:
        return (matrix[i,j+1], matrix[i+1,j], matrix[i,j-1])
    if j==0:
        return (matrix[i,j+1], matrix[i+1,j], matrix[i-1,j])
    if i==matrix_y_dim-1:
        return (matrix[i,j+1], matrix[i-1,j], matrix[i,j-1])
    if j==matrix_x_dim-1:
        return (matrix[i+1,j], matrix[i-1,j], matrix[i,j-1])

    return (matrix[i,j+1], matrix[i-1,j], matrix[i,j-1], matrix[i+1,j])


def has_neighboring_islands(i: int, j: int, matrix):
    neighboring_vertices_list = neighboring_vertices(i, j, matrix)
    return any(map(lambda x: x == 1, neighboring_vertices_list))


def island_count(matrix):
    matrix_y_dim, matrix_x_dim = matrix.shape
    island_count = 0
    for i in range(matrix_y_dim):
        for j in range(matrix_x_dim):
            if matrix[i,j] == 1: 
                island_count += 1
                if has_neighboring_islands(i, j, matrix):
                    island_count -= 1
                matrix[i,j] = -1
            print(island_count)
    return island_count

# k = np.array([[1,0,1],[0,0,0],[1,1,0],[0,1,0]])

# Binary Search
def binary_search(nums: List[int], target:int)->int:
    left = 0
    right = len(nums)-1

    while left <= right:
        mid = (left+right)//2 
        if target > nums[mid]:
            left = mid + 1
        elif target < nums[mid]:
            right = mid - 1
        else:
            return mid
    return -1


# Range Sum of a Binary Search Tree

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Tree:
    def __init__(self) -> None:
        self.leaf = None
    
    def add_node(self):
        pass


# Remove duplicates in place
def removeDuplicates(nums: List[int]) -> int:
    non_unique_counter = 0
    for i in range(len(nums)-1,0,-1):
        if nums[i] == nums[i-1]:
            del(nums[i])
    
    return len(nums)


# This doesn't work, fix it.
def moveZeroes(nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    zero_indices = []
    for i in range(len(nums)):
        if nums[i]==0:
            zero_indices.append(i)
    for i in range(len(nums)-1,-1,-1):
        if i in zero_indices:
            nums.pop(nums[i])
    nums = nums + [0 * len(zero_indices)]

def find_starting_point(arr: List[List[int]])->List[int]:
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i,j] == 1:
                return [i,j]
    return None

def flood_fill(arr: List[List[int]])->List[List[int]]:
    stack = deque()
    initial_point = find_starting_point(arr)
    stack.append(initial_point)
    while stack:
        i,j = stack.pop()
        if i<0 or i>=len(arr) or j<0 or j>=len(arr[i]):
            continue
        if arr[i,j] != 1:
            continue
        arr[i,j] = 2
        stack.appendleft([i-1, j]) # If you use append, its DFS, if you use appendleft, and queue, its BFS
        stack.appendleft([i, j-1])
        stack.appendleft([i, j+1])
        stack.appendleft([i+1, j])
    return arr


def knight_moves(initial_point, board_size)->List[List[int]]:
    
    queue = deque()
    board = np.zeros(board_size, dtype=int)
    next_move_knight_deltas = [[2,1], [2,-1], [1,2], [1,-2], [-1,2], [-1,-2], [-2,1], [-2,-1]]
    x_i, y_i = initial_point
    queue.append([x_i, y_i, 0])
    
    while queue:
        i, j, step_no = queue.pop()
        if i<0 or i>=len(board) or j<0 or j>=len(board[i]):
            continue
        if board[i, j] != 0:
            continue
        board[i, j] = step_no
        
        next_moves= [[i + delta_x, j + delta_y, step_no+1] for delta_x, delta_y in next_move_knight_deltas]
        queue.extendleft(next_moves) 

    board[x_i, y_i] = 0 # Set the initial points step number to 0.
    return board


# knight = knight_moves([32,32],[128,128])
# print(knight)


def knight_moves_on_islands(initial_point, matrix):
    queue = deque()
    board = np.zeros(matrix.shape, dtype=int)
    next_move_knight_deltas = [[2,1], [2,-1], [1,2], [1,-2], [-1,2], [-1,-2], [-2,1], [-2,-1]]
    x_i, y_i = initial_point

    if matrix[x_i, y_i] == 0:
        print("The initial point is invalid")
        return

    queue.append([x_i, y_i, 0])

    while queue:
        i, j, step_no = queue.pop()

        if i<0 or i>=len(board) or j<0 or j>=len(board[i]):
            continue
        if matrix[i,j] == 0:
            continue
        if board[i, j] != 0:
            continue
        
        board[i, j] = step_no

        next_moves= [[i + delta_x, j + delta_y, step_no+1] for delta_x,delta_y in next_move_knight_deltas]
        queue.extendleft(next_moves) 

    board[x_i, y_i] = 0 # Set the initial points step number to 0.
    return board
    
a = (np.random.random(size=[128,128])>0.5).astype('int')-1

k = knight_moves_on_islands([32,32], a)
print(k)

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
plt.title("Knight moves on islands", fontdict=font)
plt.imshow(k)

plt.colorbar(label="nth move on the island")
plt.show()



def moving_average(matrix: np.array, size:int):
    return np.convolve(matrix, np.ones(size)/size, 'full')
    # Convolution has 3 different modes: full, same, valid.

# a = np.array([-1,1,2,3,4,5,6,7,8,9])

# b = moving_average(a, 3)
# print(b)



