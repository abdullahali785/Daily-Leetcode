from collections import defaultdict

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


ans = Solution()
#print(ans.trap([0,2,0,3,1,0,1,3,2,1]))
print(ans.trap([1,5,2,3,4]))