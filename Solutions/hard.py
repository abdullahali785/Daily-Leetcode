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


ans = Solution()
#print(ans.trap([0,2,0,3,1,0,1,3,2,1]))
print(ans.trap([1,5,2,3,4]))