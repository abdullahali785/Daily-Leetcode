import collections
from typing import Optional
from collections import defaultdict
from collections import deque

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
    def trap(self, height: list[int]) -> int:
        n = len(height)
        if n == 0:
            return 0
        
        prefix = [None] * n
        suffix = [None] * n

        prefix[0] = height[0]
        for i in range(1, n):
            prefix[i] = max(prefix[i-1], height[i])
        
        suffix[n-1] = height[n-1]
        for i in range(n-2, -1, -1):
            suffix[i] = max(suffix[i+1], height[i])

        water = 0
        for i in range(n):
            water_i = (min(prefix[i], suffix[i]) - height[i]) 
            if water_i > 0:
                water += water_i 

        return water 
    
    def largestRectangleArea(self, heights: list[int]) -> int:
        maxArea = 0
        stack = []

        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                maxArea = max(maxArea, height * (i - index))
                start = index
            stack.append((start, h))

        for i, h in stack:
            maxArea = max(maxArea, h * (len(heights) - i))

        return maxArea 
    
    def minWindow(self, s: str, t: str) -> str:
        if t == "": return ""

        countT, countS = {}, {}
        for a in t:
            countT[a] = 1 + countT.get(a, 0)

        have, need = 0, len(countT)
        res, resLen = [-1, -1], float("infinity")
        l = 0

        for r in range(len(s)):
            c = s[r]
            countS[c] = 1 + countS.get(c, 0)

            if c in countT and countS[c] == countT[c]:
                have += 1

            while have == need:
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = (r - l + 1)

                countS[s[l]] -= 1
                if s[l] in countT and countS[s[l]] < countT[s[l]]:
                    have -= 1

                l += 1

        l, r = res 
        return s[l: r+1] if resLen != float("infinity") else ""

    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        res = []
        q = collections.deque()
        l = r = 0

        while r < len(nums):
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)

            if l > q[0]:
                q.popleft()

            if (r + 1) >= k:
                res.append(nums[q[0]])
                l += 1

            r += 1

        return res 

    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)
        tail = dummy 

        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next

            tail = tail.next

        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2

        return dummy.next
    
    def mergeKLists(self, lists: list[Optional[ListNode]]) -> Optional[ListNode]:
        if len(lists) == 0:
            return None
            
        for i in range(1, len(lists)):
            lists[i] = self.mergeTwoLists(lists[i], lists[i-1])

        return lists[-1]
    
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        groupPrev = dummy

        while True:
            kth = self.getKth(groupPrev, k)
            if not kth:
                break
            groupNext = kth.next

            prev, crr = kth.next, groupPrev.next
            while crr != groupNext:
                nxt = crr.next
                crr.next = prev
                prev = crr
                crr = nxt

            nxt = groupPrev.next
            groupPrev.next = kth
            groupPrev = nxt

        return dummy.next

    def getKth(self, crr, k):
        while crr and k > 0:
            crr = crr.next
            k -= 1
        return crr
            
ans = Solution()
print(ans.maxSlidingWindow([1,2,1,0,4,2,6], 3))