## 题目描述
Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0). Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

### Example 1:


Input: height = [1,8,6,2,5,4,8,3,7]


Output: 49


Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.、
## 解题思路
双指针，由两侧向内遍历。

## C++(100ms~)
```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        long long re = 0;
        int left = 0;
        int right = height.size()-1;
        int max_h = 0;
        while(left<right){
            int h = height[right]>height[left]?height[left]:height[right];
            max_h = max_h>h?max_h:h;
            long long area = h*(right-left);
            re = re>area?re:area;
            while(left<right && height[right]<=max_h){right --;}
            while(left<right && height[left]<=max_h){left ++;}
        }
        return re;
    }
};
```

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        long long re = 0;
        int left = 0;
        int right = height.size()-1;
        while(left<right){
            int h = height[right]>height[left]?height[left]:height[right];
            long long area = h*(right-left);
            re = re>area?re:area;
            if(height[left]>height[right]){right --;}
            else{left ++;}
        }
        return re;
    }
};
```
