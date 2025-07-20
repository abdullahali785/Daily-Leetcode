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
    
    

ans = Solution()
print(ans.productExceptSelf([-1,0,1,2,3]))