from collections import defaultdict

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
              
ans = Solution()
print(ans.characterReplacement("XYYX", 2))