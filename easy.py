from collections import defaultdict

class Solution:

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
    
    

ans = Solution()
print(ans.productExceptSelf([-1,0,1,2,3]))