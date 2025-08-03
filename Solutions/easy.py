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
            
    def isValid(self, s: str) -> bool:
        parentheses = {')':'(', ']':'[', '}':'{'}
        stack = []

        for a in s:
            if a in parentheses:
                if stack and stack[-1] == parentheses[a]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(a)

        return True if not stack else False 



ans = Solution()
print(ans.isValid("{([])}"))