# 17020021054
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s_len=len(s)
        temp=[]
        i=0
        j=0
        while i<s_len:
            temp.append(s[i])
            j=len(temp)-1
            if temp[j]==')':
                if temp[j-1]=='(':
                    del temp[j]
                    del temp[j-1]
                else:
                    break
                    return False
            elif temp[j]==']':
                if temp[j-1]=='[':
                    del temp[j]
                    del temp[j-1]
                else:
                    return False
            elif temp[j]=='}':
                if temp[j-1]=='{':
                    del temp[j]
                    del temp[j-1]
                else:
                    return False
            i=i+1
        if len(temp)==0:
            return True
        else:
            return False
