

我们要实现的 **`resolution_prover`（命题逻辑归结推理器）** 程序的整体目标是：

> 给定知识库（KB）和一个结论 sentence，判断是否能通过 **归结（Resolution）原理** 从 KB 推出结论。如果能推出，输出最短推理链（空子句出现的归结过程）。

------

## 程序总体思路

验证 $KB⊨α$ 是否成立，即判断知识库 KB 是否可以推出结论 α。

**逻辑等价转化：**$KB \models \alpha \iff KB \cup \{\neg \alpha\} \vdash \emptyset$

也就是说，只要从 `KB ∧ ¬α` 能推出空子句（矛盾），那么结论 α 就成立。

------

## 程序主要流程

### Step 1: 解析输入 & 语法检查

输入两行：

```
(A-B),!B
A
```

- 第一行：知识库（多个命题公式）
- 第二行：结论 sentence

我们顺序读入每一个输入token（除去空格），对每一个字句（Clause）构建出语法树（AST）。在语法树中，每一个变量（Var）是一个叶子节点，每一个运算符号（Neg, Add, Or, Imp, Iff）都是一个非叶节点。

在构建语法树的过程中，会同时进行语法检查，包括括号是否匹配，操作符是否合法（`! + - > =`），原子命题是否符合命名规则（以大写字母开头）等。若不合法 → 报错退出。

我们最后得到每一个字句（Clause）的语法树根节点。

------

### Step 2: 转换为合取范式（CNF）

每个字句 Clause 需按以下顺序转换：

| 步骤                                                         | 转换规则                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1️⃣ 消去等价和蕴含                                             | `A = B` → `(A > B) + (B > A)`<br />`A > B` → `!A - B`        |
| 2️⃣ 移动否定（得到 NNF），即仅允许否定出现在原子命题前面       | `!(A - B)` → `!A + !B`，`!(A + B)` → `!A - !B`               |
| 3️⃣ 化简为 CNF（合取范式）：一堆“子句”的合取，每个子句是一组文字的析取 | `L + R` → `nodeToCNF(L) + nodeToCNF(R)`<br />`(A1 + A2) - (B1 + B2)` → `(A1-B1)+(A1-B2)+(A2-B1)+(A2-B2)` |

------

### Step 3: 构建初始子句集

将 KB 的所有子句加入集合，再加入结论的否定`¬α`，得到`KB ∧ ¬α`的完整初始 CNF。

------

### Step 4: 归结循环（Resolution Loop）

基本规则：

> 如果两个子句中存在一对互补文字 `P` 和 `!P`，则它们可以归结生成一个新子句。

#### 算法核心伪代码：

```
clauses = KB ∪ {¬α}
new = {}

while (true):
    for each (C1, C2) in clauses:
        resolvents = resolve(C1, C2)
        if ∅ in resolvents:
            // 成功推出空子句
            return YES
        add new resolvents into new
    if new ⊆ clauses:
        // 无新子句可生成，推不出结论
        return NO
    clauses = clauses ∪ new
```

------

### Step 5：检测空子句

若某次 `resolve(C1, C2)` 结果为空集，说明矛盾产生（空子句 ∅），即 KB ⊢ α。

输出：

```
yes
N    // 最短推理步数
归结过程...
```

否则：

```
no
```

------

### 优化点与辅助逻辑

#### 去重与冗余消除

```
void add_clause_unique(vector<Clause>& clauses, Clause c)
```

功能：

- 排序并去除重复文字；
- 跳过永真子句（即包含 A 与 !A）；
- 跳过重复子句；
- 插入有效子句。

#### 防止内存移动导致引用失效

- 不在 `push_back` 后继续使用旧引用；
- 改用索引或提前拷贝。

#### 记录推理链（Parent）

每个新子句由两个父子句归结而来：

```
struct Parent { int a, b; string resolvent_repr; bool is_initial; };
```

用于回溯打印推理过程。

------

## 核心数据结构

| 名称                       | 类型                                                         | 含义                         |
| -------------------------- | ------------------------------------------------------------ | ---------------------------- |
| `Node`                     | `{ NodeType t；string name; Node *left, *right; bool neg; string var; }` | 语法树的节点                 |
| `Literal`                  | `{ string var; bool neg; }`                                  | 文字（原子或其否定）         |
| `Clause`                   | `vector<Literal>`                                            | 子句（析取）                 |
| `Parent`                   | `{ int a, b; bool is_initial; }`                             | 记录推理父节点               |
| `ClauseRecord`             | `{ Clause clause; Parent parent; Bool isInitial;}`           | 子句+来源信息                |
| `vector<ClauseRecord> all` |                                                              | 所有已生成的子句及其推理信息 |

------

## 输出示例

输入：

```
(A-B),!B
A
```

输出：（通过 3 步归结得出空子句，最后输出推理路径。）

```
yes
3
(A-B)+!B+!A
A+!A
0
```

------

## 总结：程序整体框架

```
main():
    read KB, sentence
    parse & check syntax
    convert to CNF
    add ¬sentence to KB
    resolution_prove(clauses)
        ├── resolve all pairs
        ├── add_clause_unique
        ├── check for empty clause
        ├── track parents
    if empty clause found:
        print "yes" + trace
    else:
        print "no"
```

## 编译运行命令
```bash
g++ -std=c++17 -O2 resolution_prover.cpp -o resolver
./resolver < ./测试用例/1.in
```