**剑指 Offer 03. 数组中重复的数字**

        class Solution {
        public:
            int findRepeatNumber(vector<int>& nums) {
                unordered_map<int, int> map;
                for(int i=0;i<nums.size();i++)
                {
                    if(map.find(nums[i])!=map.end()){
                        return nums[i];
                    }
                    else map[nums[i]]++;
                }
                return -1;
            }
        };

**剑指 Offer 04. 二维数组中的查找**


        class Solution {
        public:
            bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
                int i=matrix.size()-1,j=0;
                while(i>=0&&j<matrix[0].size()){
                    if(matrix[i][j]==target) return true;
                    else if(matrix[i][j]>target) i--;
                    else j++; 
                }
                return false;
            }
        };

**剑指 Offer 05. 替换空格**

        class Solution {
        public:
            string replaceSpace(string s) {
                if(s.length()==0) return s;
                int length1=s.size()-1;
                int numberOfblank=0;
                for(auto c:s){
                    if(c==' ') ++numberOfblank;
                }
                int newlength=length1+numberOfblank*2;
                int indexOfOriginal=length1;
                int indexOfNew=newlength;
                s+=string(numberOfblank*2,' ');
                while(indexOfOriginal>=0&&indexOfNew>indexOfOriginal){
                    if(s[indexOfOriginal]==' '){
                        s[indexOfNew--]='0';
                        s[indexOfNew--]='2';
                        s[indexOfNew--]='%';
                    }
                    else{
                        s[indexOfNew--]=s[indexOfOriginal];
                    }
                    --indexOfOriginal;
                }
                return s;
            }
        };

**剑指 Offer 06. 从尾到头打印链表**

        /**
        * Definition for singly-linked list.
        * struct ListNode {
        *     int val;
        *     ListNode *next;
        *     ListNode(int x) : val(x), next(NULL) {}
        * };
        */
        class Solution {
        public:
            vector<int> reversePrint(ListNode* head) {
                stack<ListNode*> nodes;
                vector<int> b;
                ListNode* pNode=head;
                while(pNode!=nullptr){
                    nodes.push(pNode);
                    pNode=pNode->next;
                }
                while(!nodes.empty()){
                    pNode=nodes.top();
                    b.push_back(pNode->val);
                    nodes.pop();
                }
                return b;
            }
        };

**剑指 Offer 07. 重建二叉树**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
                return build(preorder, inorder, 0, 0, inorder.size()-1);
            }
            TreeNode* build(vector<int>& preorder, vector<int>& inorder, int root, int start, int end){// 中序的start和end
                if(start>end) return NULL;
                TreeNode *tree =new TreeNode(preorder[root]);
                int i=start;
                while(i<end&&preorder[root]!=inorder[i]) i++;
                tree->left=build(preorder, inorder, root+1, start,i-1);
                tree->right=build(preorder, inorder, root+1+i-start, i+1, end);
                return tree;
            }
        };

**剑指 Offer 09. 用两个栈实现队列**

        class CQueue {
            stack<int> stack1,stack2;
        public:
            CQueue() {
                while(!stack1.empty()) stack1.pop();
                while(!stack2.empty()) stack2.pop();
            }
            
            void appendTail(int value) {
                stack1.push(value);
            }
            
            int deleteHead() {
                if(stack2.empty()) {
                    while(!stack1.empty()){
                        stack2.push(stack1.top());
                        stack1.pop();
                    }
                }
                if(stack2.empty()){
                    return -1;
                }else{
                    int deleteItem=stack2.top();
                    stack2.pop();
                    return deleteItem;
                }
            }
        };

        /**
        * Your CQueue object will be instantiated and called as such:
        * CQueue* obj = new CQueue();
        * obj->appendTail(value);
        * int param_2 = obj->deleteHead();
        */

**剑指 Offer 10- I. 斐波那契数列**

        class Solution {
        public:
            int fib(int N) {
                if(N<1) return 0;
                if(N==1||N==2) return 1;
                int a=1;
                int b=1;
                int res=0;
                for(int i=3;i<=N;i++){
                    res=(a+b)%1000000007;
                    a=b%1000000007;
                    b=res%1000000007;
                }
                return b;
            }
        };

**剑指 Offer 10- II. 青蛙跳台阶问题**

        class Solution {
        public:
            int fib(int N) {
                if(N<1) return 0;
                if(N==1||N==2) return 1;
                int a=1;
                int b=1;
                int res=0;
                for(int i=3;i<=N;i++){
                    res=(a+b)%1000000007;
                    a=b%1000000007;
                    b=res%1000000007;
                }
                return b;
            }
        };

**剑指 Offer 11. 旋转数组的最小数字**

        class Solution {
        public:
            int minArray(vector<int>& numbers) {
                int left=0,right=numbers.size()-1;
                while(left<=right){
                    int mid=(right+left)>>1;
                    if(numbers[left]==numbers[mid]&&numbers[mid]==numbers[right]) right--;
                    else if(numbers[mid]<=numbers[right]) right=mid;
                    else 
                        left=mid+1;
                }
                return numbers[left];
            }
        };

**剑指 Offer 12. 矩阵中的路径**

        class Solution {
        public:
            
            bool dfs(vector<vector<char>>& board, string word, int rows, int cols, int i, int j, int k){
                if(k==word.size()) return true;
                if(i<0||i>=rows||j<0||j>=cols||board[i][j]!=word[k]) return false;
                board[i][j]='#';
                if(
                    dfs(board,word,rows,cols,i - 1,j,k + 1) ||
                    dfs(board,word,rows,cols,i + 1,j,k + 1) ||
                    dfs(board,word,rows,cols,i,j - 1,k + 1) ||
                    dfs(board,word,rows,cols,i,j + 1,k + 1)
                ) return true;
            
                board[i][j] = word[k];//恢复之前的状态
                return false;
            }
            bool exist(vector<vector<char>>& board, string word) {
                int rows=board.size();
                int cols=board[0].size();
                for(int i=0;i<rows;i++){
                    for(int j=0;j<cols;j++){
                        if(dfs(board,word,rows, cols, i,j,0)) return true;
                        
                    }
                }
                return false;
            }
        };

**剑指 Offer 13. 机器人的运动范围**

        class Solution {
        public:
            int movingCount(int m, int n, int k) {
                vector<vector<int>> vec(m, vector<int>(n, 0));
                int count=0;
                dfs(0, 0, m, n, k, count, vec);
                return count;
            }

            void dfs(int row, int col, int m, int n, int k, int& count, vector<vector<int>>& vec){
                if(row<0||row>=m||col<0||col>=n) return;
                if ( row / 10 + row % 10 + col / 10 + col % 10 > k )
                    return;
                if(vec[row][col]==1) return;
                vec[row][col]=1;
                ++count;
                dfs(row+1,col,m,n,k,count,vec);
                dfs(row,col+1,m,n,k,count,vec);
            }
        };

**剑指 Offer 14- I. 剪绳子**

        class Solution {
        public:
            int cuttingRope(int length) {
                if(length<2) return 0;
                if(length==2) return 1;
                if(length==3) return 2;

                int* products = new int[length+1];
                products[0]=0;
                products[1]=1;
                products[2]=2;
                products[3]=3;

                int max=0;
                for(int i=4;i<=length;i++){
                    max=0;
                    for(int j=1;j<=i/2;j++){
                        int product=products[j]*products[i-j];
                        if(max<product) max=product;
                        products[i]=max;
                    }
                }
                max=products[length];
                delete[] products;

                return max;
            }
        };

**剑指 Offer 14- II. 剪绳子 II**

        class Solution {
        public:
            int cuttingRope(int n) {
                if(n==2) return 1;
                if(n==3) return 2;
                long ans=1;
                while(n>5){
                    ans*=3;
                    ans%=1000000007;
                    n-=3;
                }
                if(n==5) ans*=2*3;
                else if(n==4) ans*=2*2;
                else if(n==3) ans*=3; 
                ans%=1000000007;
                return ans;
            }
        };

**剑指 Offer 15. 二进制中1的个数**

        class Solution {
        public:
            int hammingWeight(uint32_t n) {
                uint32_t t=1;
                uint32_t count=0;
                while(n){
                    if(n&t) count++;
                    n>>=1;
                }
                return (int)count;
            }
        };


**剑指 Offer 16. 数值的整数次方**

        class Solution {
        public:
            double myPow(double x, int n) {
                double res=1.0;
                int t=n;
                while(n){
                    if(n&1) res *=x;
                    x*=x;
                    n/=2;
                }
                return t>0?res:1.0/res;
            }
        };

**剑指 Offer 17. 打印从1到最大的n位数**

        class Solution {
        public:
            vector<int> printNumbers(int n) {
                vector<int> res;
                int N=0;
                for(int i=0;i<n;i++){
                    N+=9*pow(10,i);
                }
                for(int i=1;i<=N;i++){
                    res.push_back(i);
                }
                return res;
            }
        };

**剑指 Offer 18. 删除链表的节点**

        /**
        * Definition for singly-linked list.
        * struct ListNode {
        *     int val;
        *     ListNode *next;
        *     ListNode(int x) : val(x), next(NULL) {}
        * };
        */
        class Solution {
        public:
            ListNode* deleteNode(ListNode* head, int val) {
                if(head==nullptr) return head;
                ListNode* cur=head;
                ListNode* pre=NULL;
                if(cur->val==val) return head->next;
                while(cur->val!=val){
                    pre=cur;
                    cur=cur->next;
                }
                pre->next=pre->next->next;
                return head;
            }
        };

**剑指 Offer 19. 正则表达式匹配**
https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/solution/dong-tai-gui-hua-chao-xiang-xi-jie-da-you-fan-ru-j/

        class Solution {
        public:
        bool isMatch(string s, string p) {
                int m = s.size(), n = p.size();
                vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
                dp[0][0] = true;
                
                for (int i = 0; i <= m; ++i) {
                    for (int j = 1; j <= n; ++j) {
                        if (p[j - 1] == '*') {
                            dp[i][j] = dp[i][j - 2] || 
                                    i && dp[i - 1][j] && 
                                    (s[i - 1] == p[j - 2] || p[j - 2] == '.');
                        } else {
                            dp[i][j] = i && dp[i - 1][j - 1] && 
                                    (s[i - 1] == p[j - 1] || p[j - 1] == '.');
                        }
                    }
                }
                return dp.back().back();
            }
        };

**剑指 Offer 20. 表示数值的字符串**

        class Solution {
        public:
            bool isNumber(string s) {
                //1、从首尾寻找s中不为空格首尾位置，也就是去除首尾空格
                int i=s.find_first_not_of(' ');
                if(i==string::npos)return false;
                int j=s.find_last_not_of(' ');
                s=s.substr(i,j-i+1);
                if(s.empty())return false;

                //2、根据e来划分底数和指数
                int e=s.find('e');

                //3、指数为空，判断底数
                if(e==string::npos)return judgeP(s);

                //4、指数不为空，判断底数和指数
                else return judgeP(s.substr(0,e))&&judgeS(s.substr(e+1));
            }

            bool judgeP(string s)//判断底数是否合法
            {
                bool result=false,point=false;
                int n=s.size();
                for(int i=0;i<n;++i)
                {
                    if(s[i]=='+'||s[i]=='-'){//符号位不在第一位，返回false
                        if(i!=0)return false;
                    }
                    else if(s[i]=='.'){
                        if(point)return false;//有多个小数点，返回false
                        point=true;
                    }
                    else if(s[i]<'0'||s[i]>'9'){//非纯数字，返回false
                        return false;
                    }
                    else{
                        result=true;
                    }
                }
                return result;
            }

            bool judgeS(string s)//判断指数是否合法
            {   
                bool result=false;
                //注意指数不能出现小数点，所以出现除符号位的非纯数字表示指数不合法
                for(int i=0;i<s.size();++i)
                {
                    if(s[i]=='+'||s[i]=='-'){//符号位不在第一位，返回false
                        if(i!=0)return false;
                    }
                    else if(s[i]<'0'||s[i]>'9'){//非纯数字，返回false
                        return false;
                    }
                    else{
                        result=true;
                    }
                }
                return result;
            }
        };

**剑指 Offer 21. 调整数组顺序使奇数位于偶数前面**

        class Solution {
        public:
            vector<int> exchange(vector<int>& nums) {
                int n=nums.size();
                int i=0,j=n-1;
                while(i<j){
                    while(nums[i]%2==1&&i<j) i++;
                    while(nums[j]%2==0&&i<j) j--;
                    swap(nums[i],nums[j]);
                }
                return nums;
            }
        };

**剑指 Offer 22. 链表中倒数第k个节点**

        /**
        * Definition for singly-linked list.
        * struct ListNode {
        *     int val;
        *     ListNode *next;
        *     ListNode(int x) : val(x), next(NULL) {}
        * };
        */
        class Solution {
        public:
            ListNode* getKthFromEnd(ListNode* head, int k) {
                // vector<ListNode*> res;
                // while(head){
                //     res.push_back(head);
                //     head=head->next;
                // }
                // return res[res.size()-k];
                ListNode* fast=head;
                ListNode* slow=head;


                for(int i=0;i<k&&fast;++i){
                    fast=fast->next;
                }

                while(fast){
                    fast=fast->next;
                    //cout<<fast->val<<endl;
                    slow=slow->next;
                }
                //cout<<slow->val<<endl;
                
                return slow;

            }
        };

**剑指 Offer 24. 反转链表**

        /**
        * Definition for singly-linked list.
        * struct ListNode {
        *     int val;
        *     ListNode *next;
        *     ListNode(int x) : val(x), next(NULL) {}
        * };
        */
        class Solution {
        public:
            ListNode* reverseList(ListNode* head) {
                ListNode* pre=NULL;
                ListNode* cur=head;
                while(cur){
                    ListNode* temp = cur->next;
                    cur->next=pre;
                    pre=cur;
                    cur=temp;
                }
                return pre;
            }
        };

**剑指 Offer 25. 合并两个排序的链表**

        /**
        * Definition for singly-linked list.
        * struct ListNode {
        *     int val;
        *     ListNode *next;
        *     ListNode(int x) : val(x), next(NULL) {}
        * };
        */
        class Solution {
        public:
            ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
                ListNode* preHead = new ListNode(-1);

                ListNode* prev = preHead;
                while (l1 != nullptr && l2 != nullptr) {
                    if (l1->val < l2->val) {
                        prev->next = l1;
                        l1 = l1->next;
                    } else {
                        prev->next = l2;
                        l2 = l2->next;
                    }
                    prev = prev->next;
                }
                prev->next = l1 == nullptr ? l2 : l1;

                return preHead->next;
            }
            
        };

**剑指 Offer 26. 树的子结构**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            bool isSubStructure(TreeNode* A, TreeNode* B) {
                if(A==nullptr||B==nullptr){
                    return false;
                }
                return hasSubStructure(A, B)||isSubStructure(A->left,B)||isSubStructure(A->right,B);
            }
            bool hasSubStructure(TreeNode* A, TreeNode* B){
                if(B==nullptr) return true;
                if(A==nullptr) return false;
                if(A->val!=B->val) return false;
                return hasSubStructure(A->left, B->left)&&hasSubStructure(A->right, B->right); 
            }
        };

**剑指 Offer 27. 二叉树的镜像**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            TreeNode* mirrorTree(TreeNode* root) {
                if(!root) return root;
                TreeNode* tmp=root->left;
                root->left = root->right;
                root->right =tmp;
                mirrorTree(root->left);
                mirrorTree(root->right);
                return root;
            }
        };
    
**剑指 Offer 28. 对称的二叉树**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        bool check(TreeNode* p, TreeNode* q){
            if(p==nullptr&&q==nullptr) return true;
            if(p==nullptr||q==nullptr||p->val!=q->val) return false;
            return check(p->left,q->right) && check(p->right,q->left);
        }
        public:
            bool isSymmetric(TreeNode* root) {
                if(root==nullptr) return true;
                return check(root->left, root->right);


            }
        };

**剑指 Offer 29. 顺时针打印矩阵**

        class Solution {
        public:
            vector<int> spiralOrder(vector<vector<int>>& matrix) {
                vector<int> res;
                if(matrix.empty()) return res;
                int rl=0, rh=matrix.size()-1;
                int cl=0, ch=matrix[0].size()-1;
                while(1){
                    for(int i=cl;i<=ch;i++) res.push_back(matrix[rl][i]);
                    if(++rl>rh) break;
                    for(int i=rl;i<=rh;i++) res.push_back(matrix[i][ch]);
                    if(--ch<cl) break;
                    for(int i=ch;i>=cl;i--) res.push_back(matrix[rh][i]);
                    if(--rh<rl) break;
                    for(int i=rh;i>=rl;i--) res.push_back(matrix[i][cl]);
                    if(++cl>ch) break;
                }
                return res;
            }
        };

**剑指 Offer 30. 包含min函数的栈**


        class MinStack {
        public:
            stack<int> A;
            stack<int> B;
            /** initialize your data structure here. */
            MinStack() {

            }
            
            void push(int x) {
                A.push(x);
                if(B.empty()||B.top()>x) B.push(x);
            }
            
            void pop() {
                if(A.top()==B.top()) B.pop();
                A.pop();
            }
            
            int top() {
                return A.top();
            }
            
            int min() {
                return B.top();
            }
        };

        /**
        * Your MinStack object will be instantiated and called as such:
        * MinStack* obj = new MinStack();
        * obj->push(x);
        * obj->pop();
        * int param_3 = obj->top();
        * int param_4 = obj->min();
        */

**剑指 Offer 31. 栈的压入、弹出序列**

        class Solution {
        public:
            bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
                stack<int> st;
                int cur=0;
                for(int i=0;i<pushed.size();i++){
                    st.push(pushed[i]);
                    while(!st.empty()&&st.top()==popped[cur])
                    {
                        st.pop();
                        ++cur;
                    }
                }
                return st.empty();
            }
        };

**剑指 Offer 32 - I. 从上到下打印二叉树**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            vector<int> levelOrder(TreeNode* root) {
                queue<TreeNode*>q;
                vector<int>res;
                if(root)q.push(root);
                while(!q.empty()){
                    int q_size = q.size();
                    for(int i = 0; i < q_size; i++){
                        TreeNode* N = q.front();
                        q.pop();
                        res.push_back(N->val);
                        if(N->left)q.push(N->left);
                        if(N->right)q.push(N->right);
                    }
                }
                return res;
            }
        };

**剑指 Offer 32 - II. 从上到下打印二叉树 II**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            vector<vector<int>> levelOrder(TreeNode* root) {
                if(!root) return {};
                vector<vector<int>> res;
                queue<TreeNode*> q;
                q.push(root);
                while(q.size()){
                    int len=q.size();
                    vector<int> level;
                    for(int i=0;i<len;i++){
                        TreeNode* t=q.front();
                        q.pop();
                        level.push_back(t->val);
                        if(t->left) q.push(t->left);
                        if(t->right) q.push(t->right);
                    }
                    res.push_back(level);
                }
                return res;
            }
        };

**剑指 Offer 32 - III. 从上到下打印二叉树 III**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            vector<vector<int>> levelOrder(TreeNode* root) {
                vector<vector<int>> res;
                queue<TreeNode*> q;
                if(root) q.push(root);
                int level_num=0;
                while(q.size())
                {
                    int q_size=q.size();
                    deque<int> dq;
                    for(int i=0;i<q_size;i++)
                    {
                        TreeNode* t=q.front();
                        q.pop();
                        if(level_num%2==0) dq.push_back(t->val);
                        else dq.push_front(t->val);
                        if(t->left) q.push(t->left);
                        if(t->right) q.push(t->right);

                    }
                    res.push_back(vector<int>(dq.begin(),dq.end()));
                    level_num++;
                }
                return res;
            }
        };

**剑指 Offer 33. 二叉搜索树的后序遍历序列**

        class Solution {
        public:
            bool verifyPostorder(vector<int>& postorder) {
                stack<int> st;
                int pre = INT_MAX;
                for(int i=postorder.size()-1;i>=0;i--){
                    if(postorder[i]>pre) return false;
                    while(st.size() && postorder[i]<st.top()){
                        pre=st.top();
                        st.pop();
                    }
                    st.push(postorder[i]);
                }
                return true;
            }
        };

**剑指 Offer 34. 二叉树中和为某一值的路径**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            vector<vector<int>> all_path;
            void find_path(TreeNode* root, int sum, vector<int>& sub_path){
                if(root==nullptr){
                    return;
                }
                sub_path.emplace_back(root->val);
                if(sum==root->val&&root->left==nullptr&&root->right==nullptr){
                    all_path.emplace_back(sub_path);
                }
                find_path(root->left,sum-root->val,sub_path);
                find_path(root->right,sum-root->val,sub_path);
                sub_path.pop_back();
            }
            
            vector<vector<int>> pathSum(TreeNode* root, int sum) {
                if(root==nullptr) return all_path;
                vector<int> sub_path;
                find_path(root, sum, sub_path);
                return all_path;
            }
        };

**剑指 Offer 35. 复杂链表的复制**

        /*
        // Definition for a Node.
        class Node {
        public:
            int val;
            Node* next;
            Node* random;
            
            Node(int _val) {
                val = _val;
                next = NULL;
                random = NULL;
            }
        };
        */
        class Solution 
        {
        public:
            Node* copyRandomList(Node* head) 
            {
                //key 旧节点
                //value 新节点
                unordered_map<Node*,Node*> m;
                Node* phead = head;
                
                while(phead != nullptr)
                {
                    m[phead] = new Node(phead->val);
                    phead = phead->next;
                }

                phead = head;

                while(phead != nullptr)
                {
                    m[phead]->next = m[phead->next];
                    m[phead]->random = m[phead->random];
                    phead = phead->next;
                }
                return m[head];
            }
        };

**剑指 Offer 36. 二叉搜索树与双向链表**

        /*
        // Definition for a Node.
        class Node {
        public:
            int val;
            Node* left;
            Node* right;

            Node() {}

            Node(int _val) {
                val = _val;
                left = NULL;
                right = NULL;
            }

            Node(int _val, Node* _left, Node* _right) {
                val = _val;
                left = _left;
                right = _right;
            }
        };
        */
        class Solution {
        public:
            Node* treeToDoublyList(Node* root) {
                if(!root) return nullptr;
                Node* head = nullptr, *pre = nullptr;
                helper(root, head, pre);
                head->left = pre;
                pre->right = head;
                return head;
            }
            void helper(Node* root, Node*& head, Node*& pre) {
                if(!root)  return;
                helper(root->left, head, pre);
                if(!head) {
                    head = root;   // 找到head
                    pre = root;    // 对pre进行初始化
                } else {
                    pre->right = root;
                    root->left = pre;
                    pre = root;
                }
                helper(root->right, head, pre);
            }
        };

**剑指 Offer 37. 序列化二叉树**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Codec {
        public:

            // Encodes a tree to a single string.
            string serialize(TreeNode* root) {
                ostringstream out;
                queue<TreeNode*> q;
                q.push(root);
                while(!q.empty()){
                    TreeNode* tmp=q.front();
                    q.pop();
                    if(tmp){
                        out<<tmp->val<<" ";
                        q.push(tmp->left);
                        q.push(tmp->right);
                    }else{
                        out<<"null ";
                    }
                }
                return out.str();
            }

            // Decodes your encoded data to tree.
            TreeNode* deserialize(string data) {
                istringstream input(data);
                string val;
                vector<TreeNode*>vec;
                while(input>>val){
                    if(val=="null"){
                        vec.push_back(NULL);
                    }else{
                        vec.push_back(new TreeNode(stoi(val)));
                    }
                }
                int j=1;
                for(int i=0;j<vec.size();i++){
                    if(vec[i]==NULL) continue;
                    if(j<vec.size()) vec[i]->left=vec[j++];
                    if(j<vec.size()) vec[i]->right=vec[j++];
                }
                return vec[0];
            }
        };

        // Your Codec object will be instantiated and called as such:
        // Codec codec;
        // codec.deserialize(codec.serialize(root));

**剑指 Offer 38. 字符串的排列**

        class Solution {
        public:
            vector<string> permutation(string s) {
                vector<string> res;
                dfs(res, s, 0);
                return res;
            }

            void dfs(vector<string> &res, string &s, int pos){
                if(pos==s.size()) res.push_back(s);
                for(int i=pos;i<s.size();i++){
                    bool flag = true;
                    for(int j = pos;j<i;j++)//字母相同时，等效，剪枝
                        if(s[j] == s[i])
                            flag = false;
                    if(flag){
                        swap(s[pos],s[i]);
                        dfs(res,s,pos+1);
                        swap(s[pos],s[i]);

                    }
                }
            }
        };

**剑指 Offer 39. 数组中出现次数超过一半的数字**

        class Solution {
        public:
            int majorityElement(vector<int>& nums) {
                unordered_map<int, int> map;
                int res;
                for(auto i:nums){
                    map[i]++;
                    if(map[i]>nums.size()/2) res= i;
                }
                return res;
            }
        };

**剑指 Offer 40. 最小的k个数**

        class Solution {
        public:
            vector<int> getLeastNumbers(vector<int>& arr, int k) {
                sort(arr.begin(), arr.end());
                vector<int> res;
                for(int i=0;i<k;i++){
                    res.push_back(arr[i]);
                }
                return res;
            }
        };


**剑指 Offer 41. 数据流中的中位数**


        class Solution{
        public:
            //大小根堆
            //小根堆 较大的数字 最小的是根节点
            //大根堆 较小的数字 最大的是根节点
            //大根堆-小根堆=1/0
            priority_queue<int> maxheap;
            priority_queue<int, vector<int>, greater<int>> minheap;

            void Insert(int num)
            {
                //都无脑先插入大根堆
                maxheap.push(num);

                //大根堆-小根堆=1/0
                if(maxheap.size()-minheap.size()>1)
                {
                    //从大根堆中拿最大的元素到小根堆
                    minheap.push(maxheap.top());
                    maxheap.pop();
                }

                //插入的元素较大
                //大根堆的根>小根堆的根
                //交换
                while(minheap.size()&&maxheap.top()>minheap.top())
                {
                    int max=maxheap.top(),min=minheap.top();
                    maxheap.pop(),minheap.pop();
                    maxheap.push(min),minheap.push(max);
                }

            }
            double GetMedian()
            {
                //奇数
                if((maxheap.size()+minheap.size())%2==1) return maxheap.top();
                return (maxheap.top()+minheap.top())/2.0;
            }
            

        }
 
**剑指 Offer 42. 连续子数组的最大和**

        class Solution {
        public:
            int maxSubArray(vector<int>& nums) {
                if(nums.size()==0) return 0;
                int sum=0;
                int maxSum=INT_MIN;
                for(auto i:nums){
                    sum+=i;
                    maxSum=max(sum,maxSum);
                    sum=sum<0?0:sum;
                }
                return maxSum;
            }
        };

**剑指 Offer 43. 1～n整数中1出现的次数**

        class Solution {
        public:
                int countDigitOne(int n) {
                int count = 0;
                for(long pos = 1;pos<=n;pos*=10){
                    int big = n/pos;
                    int small = n%pos;
                    if(big %10 ==1){
                        count +=small+1;
                    }
                    // 之所以这样写，是把第二种和第三种合在了一起
                    // 如果因为如果大于2，加8一定会进一位，如果小于等于，就算+8，也不会产生影响
                    count+=(big+8)/10 * pos;
                }
                return count;
            }
        };

**剑指 Offer 44. 数字序列中某一位的数字**

        class Solution {
        public:
            int findNthDigit(int n) 
            {
                if (n <= 9)
                    return n;
                n -= 9;
                long long count = 90, dig = 2;
                //计算数位
                while (n > count * dig)
                {
                    n -= (count * dig);
                    count *= 10;
                    dig++;
                }
                //寻找对应的那个数字
                long long num = pow(10, dig - 1) + n / dig;
                //如果刚好这个数字是在最后一位那就是上一个数字的最后一位 例如n 11 计算出来按道理是11，其实对应的是10的0
                if (n % dig == 0)
                {
                    num--;
                    return num % 10;
                }
                else
                { //如果是这个数的第二位例如 7888 那么应该78/100%10
                    for (int i = 0; i < (dig - n % dig); i++)
                    {
                        num /= 10;
                    }
                    return num % 10;
                }
            }
        };

**剑指 Offer 45. 把数组排成最小的数**

        class Solution {
        public:
            string minNumber(vector<int>& nums) {
                vector<string> strs;
                string ans;
                for(int i=0;i<nums.size();i++){
                    strs.push_back(to_string(nums[i]));
                }
                sort(strs.begin(),strs.end(),[](string& s1, string& s2){return s1+s2<s2+s1;});
                for(int i=0;i<strs.size();i++)
                    ans+=strs[i];
                return ans;
            }
        };

**剑指 Offer 46. 把数字翻译成字符串**

        class Solution {
        public:
            int translateNum(int num) {
                string str=to_string(num);
                int dp[11];
                dp[0]=1;
                dp[1]=1;
                for(int i=1;i<str.size();i++){
                    if(str[i-1]=='0'||str.substr(i-1,2)>"25"){
                        dp[i+1]=dp[i];
                    }else{
                        dp[i+1]=dp[i]+dp[i-1];
                    }
                }
                return dp[str.size()];
            }
        };

**剑指 Offer 47. 礼物的最大价值**

        class Solution {
        public:
            int maxValue(vector<vector<int>>& grid) {
                // int row=grid.size(),col=grid[0].size();
                // if(grid.empty()||grid[0].empty()) return 0;
                // vector<vector<int>> dp(row, vector<int>(col,0));
                // dp[0][0]=grid[0][0];
                // for(int i=1;i<row;i++) dp[i][0]=dp[i-1][0]+grid[i][0];
                // for(int j=1;j<col;j++) dp[0][j]=dp[0][j-1]+grid[0][j];
                // for(int i=1;i<row;i++){
                //     for(int j=1;j<col;j++){
                //         dp[i][j] = max(dp[i-1][j],dp[i][j-1])+grid[i][j];
                //     }
                // }
                // return dp[row-1][col-1];
                int row=grid.size(),col=grid[0].size();
                if(grid.empty()||grid[0].empty()) return 0;
                vector<int> dp(grid[0].begin(),grid[0].end());
                for(int j=1; j<col; j++) {
                    dp[j] += dp[j-1];
                }
                for(int i=1;i<row;i++){
                    dp[0]+=grid[i][0];
                    for(int j=1;j<col;j++){
                        dp[j] = max(dp[j-1], dp[j]) + grid[i][j];
                    }
                }
                return dp[col-1];
            }
        };

**剑指 Offer 48. 最长不含重复字符的子字符串**

        class Solution {
        public:
            int lengthOfLongestSubstring(string s) {
                unordered_set<char> occ;
                int n=s.size();
                int rk=-1, ans=0;
                for(int i=0;i<n;i++){
                    if(i!=0) occ.erase(s[i-1]);
                    while(rk+1<n&&!occ.count(s[rk+1])){
                        occ.insert(s[rk+1]);
                        ++rk;
                    }
                    ans=max(ans,rk-i+1);
                }
                return ans;
            }
        };

**剑指 Offer 49. 丑数**

        class Solution {
        public:
            int nthUglyNumber(int n) {
                if(!n) return 0;
                vector<int>ugly(n,0);
                ugly[0] = 1;     //基础丑数为1
                int i=0,j=0,k=0;  //初始分别指向三个有序链表第一个元素,这三个有序链表是想象出来的，分别就是ugly数组元素分别乘以2,3,5得到的
                for(int idx=1;idx<n;idx++)
                {
                    int tmp =  min(ugly[i]*2,min(ugly[j]*3,ugly[k]*5));    
                    //三个链表可能有相同元素，所以只要是最小的，都要移动指针
                    if(tmp == ugly[i]*2)i++;
                    if(tmp == ugly[j]*3)j++;
                    if(tmp == ugly[k]*5)k++;
                    ugly[idx] = tmp;
                }
                return ugly[n-1];
            }
        };

**剑指 Offer 50. 第一个只出现一次的字符**

        class Solution {
        public:
            char firstUniqChar(string s) {
                unordered_map<char, int> map;
                for(auto c: s)
                {
                    map[c]++;
                }
                for(auto c: s)
                {
                    if(map[c]==1) return c;
                }
                return ' ';
            }
        };

**剑指 Offer 51. 数组中的逆序对**

        class Solution {
        public:
            int mergeSort(vector<int>& nums, vector<int>& tmp, int l, int r) {
                if (l >= r) {
                    return 0;
                }

                int mid = (l + r) / 2;
                int inv_count = mergeSort(nums, tmp, l, mid) + mergeSort(nums, tmp, mid + 1, r);
                int i = l, j = mid + 1, pos = l;
                while (i <= mid && j <= r) {
                    if (nums[i] <= nums[j]) {
                        tmp[pos] = nums[i];
                        ++i;
                        inv_count += (j - (mid + 1));
                    }
                    else {
                        tmp[pos] = nums[j];
                        ++j;
                    }
                    ++pos;
                }
                for (int k = i; k <= mid; ++k) {
                    tmp[pos++] = nums[k];
                    inv_count += (j - (mid + 1));
                }
                for (int k = j; k <= r; ++k) {
                    tmp[pos++] = nums[k];
                }
                copy(tmp.begin() + l, tmp.begin() + r + 1, nums.begin() + l);
                return inv_count;
            }

            int reversePairs(vector<int>& nums) {
                int n = nums.size();
                vector<int> tmp(n);
                return mergeSort(nums, tmp, 0, n - 1);
            }
        };

// 作者：LeetCode-Solution
// 链接：https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/solution/shu-zu-zhong-de-ni-xu-dui-by-leetcode-solution/
// 来源：力扣（LeetCode）
// 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

**剑指 Offer 52. 两个链表的第一个公共节点**

        /**
        * Definition for singly-linked list.
        * struct ListNode {
        *     int val;
        *     ListNode *next;
        *     ListNode(int x) : val(x), next(NULL) {}
        * };
        */
        class Solution {
        public:
            ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
                ListNode *PA=headA;
                ListNode *PB=headB;
                while(PA != PB){
                    PA=PA!=NULL?PA->next:headB;
                    PB=PB!=NULL?PB->next:headA;
                }
                return PA;
            }
        };

**剑指 Offer 53 - I. 在排序数组中查找数字 I**

        class Solution {
        public:
            int search(vector<int>& nums, int target) {
                int left1 = 0, right1 = nums.size();
                int left2 = 0, right2 = nums.size();
                while(left1 < right1)
                {
                    int mid = left1 + (right1 - left1)/2;
                    if(nums[mid] >= target) right1 = mid;
                    else left1 = mid + 1;
                }
                while(left2 < right2)
                {
                    int mid = left2 + (right2 - left2)/2;
                    if(nums[mid] <= target) left2 = mid + 1;
                    else right2 = mid;
                }
                if(left2 >= left1) return left2 - left1;
                return 0;
            }
        };

**剑指 Offer 53 - II. 0～n-1中缺失的数字**


        class Solution {
        public:
            int missingNumber(vector<int>& nums) {
                int left = 0, right = nums.size();
                while(left < right)
                {
                    int mid = left + (right - left)/2;
                    if(nums[mid] == mid) left = mid + 1;
                    else right = mid; 
                }
                return left;
            }
        };

**剑指 Offer 54. 二叉搜索树的第k大节点**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            void dfs(TreeNode* root, int& k, int& res){
                if(k==0||root==nullptr) return;
                dfs(root->right,k,res);
                --k;
                if(k==0){
                    res=root->val;
                    return;
                }
                dfs(root->left,k,res);
            }
            int kthLargest(TreeNode* root, int k) {
                int res=0;
                dfs(root,k,res);
                return res;
            }
        };

**剑指 Offer 55 - I. 二叉树的深度**


        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            int maxDepth(TreeNode* root) {
                if(root==nullptr) return 0;
                int left=maxDepth(root->left);
                int right=maxDepth(root->right);
                return max(left,right)+1;
            }
        };

**剑指 Offer 56 - I. 数组中数字出现的次数**

        class Solution {
        public:
            vector<int> singleNumbers(vector<int>& nums) {
                unordered_map<int, int> map;
                for(auto c:nums){
                    map[c]++;
                }
                vector<int> vec;
                for(auto c:nums){
                    if(map[c]==1) vec.push_back(c);
                }
                return vec;
                
            }
        };

        class Solution {
        public:
            vector<int> singleNumbers(vector<int>& nums) {
                int s=0;
                for(int num:nums){
                    s^=num;
                }
                int k=s&(-s);
                vector<int> rs(2,0);
                for(int num:nums){
                    if(num&k){
                        rs[0]^=num;
                    }else{
                        rs[1]^=num;
                    }
                }
                return rs;
            }
        };

**剑指 Offer 56 - II. 数组中数字出现的次数 II**

        class Solution {
        public:
            int singleNumber(vector<int>& nums) {
                        int a = 0, b = 0;
                        for (auto num : nums)
                        {
                            a = (a ^ num) & ~b;
                            b = (b ^ num) & ~a;
                        }
                        return a;
            }
        };

**剑指 Offer 57. 和为s的两个数字**

        class Solution {
        public:
            vector<int> twoSum(vector<int>& nums, int target) {
                int left=0,right=nums.size()-1,sum=0;
                vector<int> res(2,0);
                while(left<right)
                {
                    sum=nums[left]+nums[right];
                    if(sum>target) right--;
                    else if(sum<target) left++;
                    else
                    {
                        res[0]=nums[left];
                        res[1]=nums[right];
                        break;
                    } 
                }
                return res;

            }
        };


**剑指 Offer 57 - II. 和为s的连续正数序列**

        class Solution {
        public:
            vector<vector<int>> findContinuousSequence(int target) {
                int i=1;
                int j=1;
                int sum=0;
                vector<vector<int>> res;

                while(i<=target/2){
                    if(sum<target){
                        sum+=j;
                        j++;
                    }else if(sum>target){
                        sum-=i;
                        i++;
                    }else{
                        vector<int> arr;
                        for(int k=i;k<j;k++){
                            arr.push_back(k);
                        }
                        res.push_back(arr);
                        sum-=i;
                        i++;
                    }

                }
                return res;
            }
        };

**剑指 Offer 58 - I. 翻转单词顺序**

class Solution {
public:
    string reverseWords(string s) {
        string res,temp;
        stringstream ss(s);
        while(ss>>temp){
            res=temp+" "+res;
        }
        return res.substr(0,res.size()-1);
    }
};

**剑指 Offer 58 - II. 左旋转字符串**

        class Solution {
        public:
            string reverseLeftWords(string s, int n) {
                return s.substr(n, s.size() - n) + s.substr(0, n);

            }
        };

**剑指 Offer 59 - I. 滑动窗口的最大值**

        class Solution {
        public:
            vector<int> maxSlidingWindow(vector<int>& nums, int k) {
                int len=nums.size();
                vector<int> vec;
                if(k==0||len==0) return vec;
                for(auto it=nums.begin();it+k!=nums.end()+1;it++){
                    int max_val=*max_element(it,it+k);
                    vec.push_back(max_val);
                }
                return vec;
            }
        };

        class Solution {
        public:
            class MyQueue { //单调队列（从大到小）
            public:
                deque<int> que; // 使用deque来实现单调队列
                void pop(int value) {
                    if (!que.empty() && value == que.front()) {
                        que.pop_front();
                    }
                }
                void push(int value) {
                    while (!que.empty() && value > que.back()) {
                        que.pop_back();
                    }
                    que.push_back(value);
                }
                int front() {
                    return que.front();
                }
            };
            vector<int> maxSlidingWindow(vector<int>& nums, int k) {
                MyQueue que;
                vector<int> result;
                for (int i = 0; i < k; i++) { // 先将前k的元素放进队列
                    que.push(nums[i]);
                }
                result.push_back(que.front()); // result 记录前k的元素的最大值
                for (int i = k; i < nums.size(); i++) {
                    que.pop(nums[i - k]); // 模拟滑动窗口的移动
                    que.push(nums[i]); // 模拟滑动窗口的移动
                    result.push_back(que.front()); // 记录对应的最大值
                }
                return result;
            }
        };

**剑指 Offer 59 - II. 队列的最大值**

        class MaxQueue {
            queue<int> q;
            deque<int> d;
        public:
            MaxQueue() {
            }
            int max_value() {
                if(d.empty()) return -1;
                return d.front();
            }
            
            void push_back(int value) {
                while(!d.empty()&&d.back()<value){
                    d.pop_back();
                }
                d.push_back(value);
                q.push(value);
            }
            
            int pop_front() {
                if (q.empty())
                    return -1;
                int ans = q.front();
                if (ans == d.front()) {
                    d.pop_front();
                }
                q.pop();
                return ans;
            }
        };

        /**
        * Your MaxQueue object will be instantiated and called as such:
        * MaxQueue* obj = new MaxQueue();
        * int param_1 = obj->max_value();
        * obj->push_back(value);
        * int param_3 = obj->pop_front();
        */

**剑指 Offer 60. n个骰子的点数**

        class Solution {
        public:
            vector<double> twoSum(int n) {
                //n <= 11
                vector<vector<double>>dp(n + 1, vector<double>(6*n + 1, 0));
                vector<double> ans;
                for(int i = 1; i <= n; i ++){
                    for(int j = i; j <= 6*i; j ++){
                        if(i == 1) {
                            dp[i][j] = 1;
                            continue;
                        }
                        for(int k = 1; k <= 6; k ++){
                            if(j - k >= i - 1) dp[i][j] += dp[i - 1][j - k];
                        }
                    }
                }
                for(int i = n; i <= 6*n; i ++){
                    ans.push_back(dp[n][i] * pow(1.0/6, n));
                }
                return ans;
            }
        };

**剑指 Offer 61. 扑克牌中的顺子**

        class Solution {
        public:
            bool isStraight(vector<int>& nums) {
                sort(nums.begin(), nums.end());
                int i=0;
                while(i<nums.size()&&nums[i]==0) i++;
                if(i==nums.size()) return true;
                for(int j=i;j<4;j++){
                    if(nums[j]==nums[j+1]) return false;
                }
                return nums[4]-nums[i]<=4;
            }
        };

**剑指 Offer 62. 圆圈中最后剩下的数字**

        class Solution {
        public:
            int lastRemaining(int n, int m) {
                int ans=0;
                for(int i=2;i<=n;i++){
                    ans=(ans+m)%i;
                }
                return ans;
            }
        };

**剑指 Offer 63. 股票的最大利润**

        class Solution {
        public:
            int maxProfit(vector<int>& prices) {
                if(prices.size() <= 1)
                    return 0;
                int min_price = prices[0], max_price = 0;
                // 记录  买入的最小值       最大获利
                for(int i = 1; i < prices.size(); i++)
                {
                    max_price = max(max_price, prices[i] - min_price);
                    min_price = min(min_price, prices[i]);
                }
                return max_price;
            }
        };

**剑指 Offer 64. 求1+2+…+n**

        class Solution {
        public:
            int sumNums(int n) {
                n&&(n+=sumNums(n-1));
                return n;
            }
        };

**剑指 Offer 65. 不用加减乘除做加法**

        class Solution {
        public:
            int add(int a, int b) {
                while(b!=0){
                    int tmp=a^b;  //非进位和
                    b=((unsigned int)(a&b)<<1);   //进位和
                    a=tmp;
                }
                return a;
            }
        };

**剑指 Offer 66. 构建乘积数组**

        class Solution {
        public:
            vector<int> constructArr(vector<int>& a) {
                int len=a.size();
                vector<int> b(len);
                int temp=1;
                for(int i=0;i<len;i++){
                    b[i]=temp;
                    temp*=a[i];
                }
                temp=1;
                for(int i=len-1;i>=0;i--){
                    b[i]*=temp;
                    temp*=a[i];
                }
                return b;
            }
        };

**剑指 Offer 67. 把字符串转换成整数**

        class Solution {
        public:
            int strToInt(string str) {
                int len=str.length(), i=0;
                bool isMinus=false;
                long result=0;
                while(str[i]==' ') i++;
                if(isalpha(str[i])) return 0;
                else if(str[i]=='-'){
                    isMinus=true;
                    i++;
                }
                else if(str[i]=='+') i++;
                while(i<len&&isdigit(str[i])){
                    result=10*result+(str[i++]-'0');
                    if(result>INT_MAX)
                        return isMinus?INT_MIN:INT_MAX;
                }
                return isMinus?-result:result;
            }
        };

**剑指 Offer 68 - I. 二叉搜索树的最近公共祖先**

        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
                if(root==nullptr) return NULL;
                if(root->val>p->val&&root->val>q->val) return lowestCommonAncestor(root->left,p,q);
                else if(root->val<p->val&&root->val<q->val) return lowestCommonAncestor(root->right,p,q);
                else return root;
            }
        };

**剑指 Offer 68 - II. 二叉树的最近公共祖先**


        /**
        * Definition for a binary tree node.
        * struct TreeNode {
        *     int val;
        *     TreeNode *left;
        *     TreeNode *right;
        *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
        * };
        */
        class Solution {
        public:
            TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
                if(!root||root==p||root==q) return root;
                TreeNode* left=lowestCommonAncestor(root->left, p, q);
                TreeNode* right=lowestCommonAncestor(root->right,p,q);
                if(left&&right) return root;
                return left?left:right;
            }
        };