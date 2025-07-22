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
    
    def sPalindrome(self, s: str) -> bool:
        cleaned = ''.join(c for c in s if c.isalnum()).lower()
        reversed = ""

        for i in range(len(cleaned)-1, -1, -1):
            reversed += cleaned[i]

        if reversed == cleaned:
            return True
        
        return False
    

ans = Solution()
print(ans.sPalindrome("Was it a car or a cat I saw?"))