## 题目描述
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.


You may assume that each input would have exactly one solution, and you may not use the same element twice.


You can return the answer in any order.

## 解题思路
可以排序后使用双指针的方式，由两侧向内搜索。排序算法的时间复杂度为O(nlog(n)),搜索算法的时间复杂度为O(n)。

## 代码
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> temp = nums;
        sort(temp.begin(),temp.end());
        int first = 0;
        int second = temp.size()-1;
        while(1){
            if(temp[first] + temp[second] == target){
                break;
            }
            if(temp[first] + temp[second] < target){
                first ++;
            }
            else{
                second --;
            }
        }
        int a = temp[first];
        int b = temp[second];
        vector<int> re;
        for(int i = 0;i < nums.size();i++){
            if(nums[i] == a){
                re.push_back(i);
            }
            else if(nums[i] == b){
                re.push_back(i);
            }
            if(re.size() == 2){
                break;
            }
        }
     return re;
    }
};
```
