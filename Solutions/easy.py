import collections
from typing import Optional
from collections import defaultdict

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

class Solution:
    def hasDuplicate(self, nums: list[int]) -> bool:
        hashmap = {}
        for num in nums:
            if num in hashmap:
                return True
            hashmap[num] = 1
        return False

    def isAnagram(self, s: str, t: str) -> bool:
        hashmap = {}

        for char in s:
            if char in hashmap:
                hashmap[char] += 1
            else:
                hashmap[char] = 1

        for char in t:
            if char in hashmap:
                hashmap[char] -= 1
            else:
                return False

        return all(v == 0 for v in hashmap.values())
    
    def twoSum(self, numbers: list[int], target: int) -> list[int]:
        p1 = 1
        p2 = len(numbers)

        while p1 < p2:
            if numbers[p1-1] + numbers[p2-1] > target:
                p2 -= 1
            elif numbers[p1-1] + numbers[p2-1] < target:
                p1 += 1
            else:
                return [p1, p2]
            
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        hashmap = {}
        length = len(nums)

        for i in range(length):
            hashmap[nums[i]] = i

        for i in range(length):
            diff = target - nums[i]
            if diff in hashmap and hashmap[diff] != i:
                return [i, hashmap[diff]]

    def twoSum(self, nums: list[int], target: int) -> list[int]:
        hashmap = {}

        for i in range(len(nums)):
            hashmap[nums[i]] = i

        for j in range(len(nums)):
            tar = target - nums[j]
            if tar in hashmap and hashmap[tar] != j:
                return [min(hashmap[tar], j), max(hashmap[tar], j)] 
    
    def encode(self, strs: list) -> str:
        result = ''
        for i in strs:
            result += str(len(i)) + '#' + i
        return result

    def decode(self, s: str) -> list[str]:
        result, i = [], 0 #i will be at the integer which shows lenght of first string.
        while i < len(s):
            j = i #j will be on the delimiter.
            while s[j] != '#':
                j += 1
            lenght = int(s[i:j])
            result.append(s[j+1:j+1+lenght])
            i = j+1+lenght
        return result

    def sPalindrome(self, s: str) -> bool:
        cleaned = ''.join(c for c in s if c.isalnum()).lower()
        reversed = ""

        for i in range(len(cleaned)-1, -1, -1):
            reversed += cleaned[i]

        if reversed == cleaned:
            return True

        return False
    
    def isValid(self, s: str) -> bool:
        stack = []
        hashmap = {'}':'{', ']':'[', ')':'('}

        for char in s:
            if char in hashmap: # Char is a closing bracket
                if stack and hashmap[char] == stack[-1]:
                    stack.pop()
                else:
                    return False 
            else: # Char is a opening bracket 
                stack.append(char)
        
        return True if stack == [] else False
    
    def isPalindrome(self, s: str) -> bool:
        text = [char.lower() for char in s if char.isalnum()]
        l, r = 0, len(text) - 1

        while l < r:
            if text[l] != text[r]: return False
            l += 1
            r -= 1    

        return True
    
    def isPalindrome_pointer(self, s: str) -> bool:
        l, r = 0, len(s) - 1

        while r > l:
            while l < r and not self.alphaNum(s[l]):
                l += 1
            while r > l and not self.alphaNum(s[r]):
                r -= 1

            if s[l].lower() != s[r].lower():
                return False 

            l, r = l + 1, r - 1

        return True

    def alphaNum(self, char: str) -> bool:
        return (ord('A') <= ord(char) <= ord('Z') or 
        ord('a') <= ord(char) <= ord('z') or 
        ord('0') <= ord(char) <= ord('9'))

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head

        while curr:
            nex = curr.next
            curr.next = prev
            prev = curr
            curr = nex

        return prev

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        res = ListNode()
        p = res

        while list1 and list2:
            if list1.val <= list2.val:
                p.next = list1
                list1 = list1.next
            else:
                p.next = list2
                list2 = list2.next
            p = p.next

        if list1:
            p.next = list1
        else:
            p.next = list2

        return res.next 

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True

        return False 
    
    def search_helper(self, nums: list[int], target: int) -> int:
        def helper(nums: list[int], target: int, low: int, high: int) -> int:
            if low > high:
                return -1

            mid = (low + high) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                return helper(nums, target, low, mid-1)
            else:
                return helper(nums, target, mid+1, high)
                
        return helper(nums, target, 0, len(nums)-1)

    def search(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            m = (l + r) // 2
            if nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                return m
        return -1

    
    def maxProfit(self, prices: list[int]) -> int:
        l, r = 0, 1
        maxP = 0

        while r < len(prices):
            if prices[l] < prices[r]:
                profit = prices[r] - prices[l]
                maxP = max(maxP, profit)
            else:
                l = r
            r += 1
        return maxP

ans = Solution()
print(ans.reorderList(ListNode(0, ListNode(1, ListNode(2, ListNode(3, ListNode(4)))))))