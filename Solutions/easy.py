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

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        crr = head

        while crr:
            nxt = crr.next
            crr.next = prev

            prev = crr
            crr = nxt

        return prev

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)
        tail = dummy

        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next

            else:
                tail.next = list2
                list2 = list2.next

            tail = tail.next

        if list1:
            tail.next = list1
        elif list2:
            tail.next = list2

        return dummy.next

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        s, f = head, head

        while f and f.next:
            s = s.next
            f = f.next.next
            if s == f:
                return True

        return False

    def reorderList(self, head: Optional[ListNode]) -> None:
        s, f = head, head.next

        while f and f.next:
            s = s.next
            f = f.next.next

        secondHalf = s.next 
        prev = s.next = None

        while secondHalf:
            nxt = secondHalf.next
            secondHalf.next = prev
            prev = secondHalf
            secondHalf = nxt

        firstHalf, secondHalf = head, prev 

        while secondHalf:
            temp1, temp2 = firstHalf.next, secondHalf.next
            firstHalf.next = secondHalf
            secondHalf.next = temp1
            firstHalf, secondHalf = temp1, temp2


ans = Solution()
print(ans.reorderList(ListNode(0, ListNode(1, ListNode(2, ListNode(3, ListNode(4)))))))