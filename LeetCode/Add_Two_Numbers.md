## 题目描述
（English）
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.
（中文）
给你两个非空链表，表示两个非负整数。数字以相反的顺序存储，每个节点都包含一个数字。将两个数字相加，并以链表形式返回总和。

你可以确定两个数字都不是以0开头，除了0本身。

## 解题思路
根据题面，其意思为将两个倒序数字相加后的结果，进行倒序输出。
由此可见，可以先搜索到链表最深处，计算数字加和后，作为新链表的开头，然后依次递归。

## 代码
### C++
```.cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        return addBybit(l1,l2,0);
    }
 
    ListNode* addBybit(ListNode* l1,ListNode* l2,int carry){
        if(l1 == NULL) l1 = new ListNode(0);
        if(l2 == NULL) l2 = new ListNode(0);
        ListNode* l = new ListNode((l1->val+l2->val+carry)%10);
        carry = (l1->val + l2->val + carry)/10;
        if(l1->next != NULL || l2->next != NULL || carry !=0 ){
            l->next = addBybit(l1->next, l2->next, carry); 
        }
        return l;
    }
};
```
### Python
```.py
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
   
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        def addByBit(l1,l2,carry):
            if l1 == None:
                l1 = ListNode(0)
            if l2 == None:
                l2 = ListNode(0)
            l = ListNode((l1.val+l2.val+carry)%10)
            carry = (l1.val+l2.val+carry)/10
            if l1.next != None or l2.next != None or carry != 0:
                l.next = addByBit(l1.next,l2.next,carry)
            return l
    
        return addByBit(l1,l2,0)
    
```
