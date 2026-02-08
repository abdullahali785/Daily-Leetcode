import math
import collections
from typing import Optional
from collections import Counter, defaultdict

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        vals = []
        curr = self
        while curr:
            vals.append(str(curr.val))
            curr = curr.next
        return "[" + " -> ".join(vals) + "]"
    
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class nodeDouble:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = self.next = None
    
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
         
    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return min(self.stack)
    
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}

        self.left, self.right = nodeDouble(0,0), nodeDouble(0,0)
        self.left.next, self.right.prev = self.right, self.left

    def remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next, nxt.prev = nxt, prev

    def insert(self, node):
        prev, nxt = self.right.prev, self.right
        prev.next = nxt.prev = node
        node.next, node.prev = nxt, prev

    def get(self, key: int) -> int:
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = nodeDouble(key, value)
        self.insert(self.cache[key])

        if len(self.cache) > self.cap:
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]

class Solution:
    def groupAnagrams_sorting(self, strs: list[str]) -> list[list[str]]:
        res = defaultdict(list)
        
        for string in strs:
            sortedS = ''.join(sorted(string))
            res[sortedS].append(string)

        return list(res.values()) 
    
    def groupAnagrams_hashtable(self, strs: list[str]) -> list[list[str]]:
        res = defaultdict(list)
        
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            res[tuple(count)].append(s)

        return list(res.values())
    
    def groupAnagrams_frequency(self, strs: list[str]) -> list[list[str]]:
        hashmap = {}

        for string in strs:
            frequency = [0] * 26
            for letter in string:
                frequency[ord(letter) - ord("a")] += 1
            
            if tuple(frequency) not in hashmap:
                hashmap[tuple(frequency)] = []
            hashmap[tuple(frequency)].append(string)

        return list(hashmap.values())
    

    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        count = defaultdict(int)
        for num in nums:
            count[num] += 1

        n = len(nums)
        buckets = [[] for i in range(n+1)]

        for num, freq in count.items():
            buckets[freq].append(num)

        res = []
        for i in range(n, 0, -1):
            for num in buckets[i]:
                res.append(num)
                if len(res) == k:
                    return res 
                
    def topKFrequent_bucket(self, nums: list[int], k: int) -> list[int]:
        freq = Counter(nums)
        ans = []
        buckets = [[] for i in range(len(nums) + 1)]

        for num, count in freq.items():
            buckets[count].append(num)

        for i in range(len(buckets) - 1, 0, -1):
            for n in buckets[i]:
                ans.append(n)
                if len(ans) == k:
                    return ans
                
    def encode(self, strs: list[str]) -> str:
        res = ""
        for string in strs:
            res += str(len(string)) + "#" + string

        return res

    def decode(self, s: str) -> list[str]:
        res, i = [], 0

        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1

            length = int(s[i : j])
            res.append(s[j + 1 : j + 1 + length])
            i = j + 1 + length

        return res 

    def productExceptSelf(self, nums: list[int]) -> list[int]:
        n = len(nums)
        output = [1] * n

        for i in range(n):
            for m in range(n):
                if m == i:
                    continue
                else:
                    output[i] *= nums[m]

        return output 
    
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        row = [set() for i in range(9)]
        column = [set() for i in range(9)]
        box = [set() for i in range(9)]
        
        for r in range(9):
            for c in range(9):

                cell = board[r][c]
                if cell == ".":
                    continue

                if cell in row[r]:
                    return False
                row[r].add(cell)

                if cell in column[c]:
                    return False
                column[c].add(cell)

                box_index = (r // 3) * 3 + (c // 3)
                if cell in box[box_index]:
                    return False
                box[box_index].add(cell) 

        return True 
    
    def longestConsecutive(self, nums: list[int]) -> int:
        numSet = set(nums)
        longest = 0

        for num in numSet:
            if num - 1 not in numSet:
                lenght = 1 
                while num + lenght in numSet:
                    lenght += 1
                longest = max(lenght, longest)
        return longest
    
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        ans = []

        for i, a in enumerate(nums):
            if i > 0 and a == nums[i-1]:
                continue

            j, k = i+1, len(nums)-1 
            target = -a

            while j < k:
                if nums[j] + nums[k] < target:
                    j += 1
                elif nums[j] + nums[k] > target:
                    k -= 1
                else:
                    ans.append([a, nums[j], nums[k]])
                    j += 1
                    while nums[j] == nums[j-1] and j < k:
                        j += 1
        
        return ans 
    
    def maxArea(self, heights: list[int]) -> int:
        l, r = 0, len(heights)-1
        capacity = []

        while l < r:
            capacity.append((r - l) * min(heights[l], heights[r]))
            if heights[l] < heights[r]:
                l += 1
            else:
                r -= 1
            
        return max(capacity)
    
    def evalRPN(self, tokens: list[str]) -> int:
        stack = []
        operators = ['+', '-', '*', '/']

        for a in tokens:
            try:
                int(a)
                stack.append(int(a))
            except:
                i1 = int(stack.pop())
                i2 = int(stack.pop())
                if a == '+':
                    stack.append(i2 + i1)
                elif a == '-':
                    stack.append(i2 - i1)
                elif a == '*':
                    stack.append(i2 * i1)
                else:
                    stack.append(i2 / i1)

        return int(stack[0])
    
    def generateParenthesis(self, n: int) -> list[str]:
        res = []

        def backtrack(currrent, opening, closing):
            if len(currrent) == 2*n:
                res.append(currrent)
                return 
            if opening < n:
                backtrack(currrent+'(', opening+1, closing)
            if closing < opening: 
                backtrack(currrent+')', opening, closing+1)

        backtrack('', 0, 0)
        return res 
    
    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        stack = []
        res = [0] * len(temperatures)

        for i, t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                stackT, stackInd = stack.pop()
                res[stackInd] = (i - stackInd)
            stack.append([t, i])

        return res 

    def carFleet(self, target: int, position: list[int], speed: list[int]) -> int:
        pair = [[p, s] for p, s in zip(position, speed)]

        stack = []
        for p, s in sorted(pair)[::-1]:
            stack.append((target - p) / s)
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()
        return len(stack)
    
    def maxProfit(self, prices: list[int]) -> int:
        l, r = 0, 1 #Left -> Buy, Right -> Sell
        maxP = 0

        while r < len(prices):
            if prices[r] > prices[l]:
                profit = prices[r] - prices[l]
                maxP = max(maxP, profit)
            else:
                l = r
            r += 1

        return maxP if maxP > 0 else 0

    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = set()
        l = 0
        res = 0

        for r in range(len(s)):
            while s[r] in seen:
                seen.remove(s[l])
                l += 1
            seen.add(s[r])
            res = max(res, r-l + 1)

        return res 

    def characterReplacement(self, s: str, k: int) -> int:
        count = {}
        res = 0
        l = 0

        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            
            if (r - l + 1) - max(count.values()) > k:
                count[s[l]] -= 1
                l += 1

            res = max(res, (r - l + 1))

        return res 
    
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False 

        l, r = 0, len(s1)
        
        while r < len(s2)+1:
            if sorted(s2[l : r]) == sorted(s1):
                return True 
            else:
                l += 1
                r += 1

        return False 
    
    def reorderList(self, head: Optional[ListNode]) -> None:
        s, f = head, head.next

        while f and f.next:
            s = s.next
            f = f.next.next

        secondHalf = s.next 
        prev = s.next = None

        while secondHalf:
            nxt = secondHalf.next
            secondHalf.next = prev
            prev = secondHalf
            secondHalf = nxt

        firstHalf, secondHalf = head, prev 

        while secondHalf:
            temp1, temp2 = firstHalf.next, secondHalf.next
            firstHalf.next = secondHalf
            secondHalf.next = temp1
            firstHalf, secondHalf = temp1, temp2

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        first = second = dummy 

        for i in range(n+1):
            first = first.next

        while first:
            first = first.next
            second = second.next

        second.next = second.next.next
        return dummy.next
    
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        copyTable = {None : None}

        crr = head
        while crr:
            copy = Node(crr.val)
            copyTable[crr] = copy
            crr = crr.next

        crr = head
        while crr:
            copy = copyTable[crr]
            copy.next = copyTable[crr.next]
            copy.random = copyTable[crr.random]
            crr = crr.next 

        return copyTable[head]
    
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        res = ListNode()
        crr = res

        carry = 0
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0

            numSum = v1 + v2 + carry
            carry = numSum // 10
            numSum = numSum % 10

            crr.next = ListNode(numSum)
            crr = crr.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return res.next
    
    def findDuplicate(self, nums: list[int]) -> int:
        for num in nums:
            index = abs(num) -1
            if nums[index] < 0:
                return abs(num)
            nums[index] *= -1

        return -1 
    
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        rows, cols = len(matrix), len(matrix[0])

        top, bot = 0, rows - 1
        while top <= bot:
            row = (top + bot) // 2
            if target > matrix[row][-1]:
                top = row + 1
            elif target < matrix[row][0]:
                bot = row - 1
            else:
                break

        if not (top <= bot):
            return False

        row = (top + bot) // 2
        l, r = 0, cols - 1
        while l <= r:
            mid = (l + r) // 2
            if target > matrix[row][mid]:
                l = mid + 1
            elif target < matrix[row][mid]:
                r = mid - 1
            else:
                return True 
        
        return False 
    
    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        l, r = 1, max(piles)
        res = r

        while l <= r:
            mid = (l + r) // 2
            hours = 0
            for p in piles:
                hours += math.ceil(p / mid)

            if hours <= h:
                res = min(res, mid)
                r = mid-1

            else:
                l = mid+1
        return res 
    
    def findMin(self, nums: list[int]) -> int:
        res = nums[0]
        l, r = 0, len(nums) - 1

        while l <= r:
            if nums[l] <= nums[r]:
                res = min(res, nums[l])
                break

            mid = (l + r) // 2
            res = min(res, nums[mid])

            if nums[l] <= nums[mid]:
                l = mid + 1
            else:
                r = mid - 1

        return res

ans = Solution()
print(ans.findDuplicate([1,2,2,3,4]))