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


ans = Solution()
#print(ans.trap([0,2,0,3,1,0,1,3,2,1]))
print(ans.trap([1,5,2,3,4]))