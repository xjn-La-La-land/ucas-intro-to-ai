// resolution_prover.cpp
// 编译: g++ -std=c++17 -O2 resolution_prover.cpp -o resolver
// 运行: ./resolver < input.txt
//
// 输入假设：第一行是知识库（若有多子句用英文逗号分隔），第二行是结论句子
// 输出参见题目说明与样例格式

#include <bits/stdc++.h>
using namespace std;

/*
 Parser + AST for propositional logic with operators:
  - ! (not)
  - + (and)
  - - (or)
  - > (implication)
  - = (biconditional)
 Atoms: start with uppercase letter, followed by letters/digits
*/

enum NodeType { VAR, NOT, AND, OR, IMP, IFF };

struct Node {
    NodeType t;
    string name; // for VAR
    shared_ptr<Node> l, r;
    Node(NodeType _t=VAR): t(_t){}
    static shared_ptr<Node> Var(const string &s){ auto p=make_shared<Node>(VAR); p->name=s; return p;}
    static shared_ptr<Node> Not(shared_ptr<Node> a){ auto p=make_shared<Node>(NOT); p->l=a; return p;}
    static shared_ptr<Node> And(shared_ptr<Node> a, shared_ptr<Node> b){ auto p=make_shared<Node>(AND); p->l=a; p->r=b; return p;}
    static shared_ptr<Node> Or(shared_ptr<Node> a, shared_ptr<Node> b){ auto p=make_shared<Node>(OR); p->l=a; p->r=b; return p;}
    static shared_ptr<Node> Imp(shared_ptr<Node> a, shared_ptr<Node> b){ auto p=make_shared<Node>(IMP); p->l=a; p->r=b; return p;}
    static shared_ptr<Node> Iff(shared_ptr<Node> a, shared_ptr<Node> b){ auto p=make_shared<Node>(IFF); p->l=a; p->r=b; return p;}
};

string input;
size_t posi;
string token;
enum TokType {TOK_END, TOK_LP, TOK_RP, TOK_NOT, TOK_AND, TOK_OR, TOK_IMP, TOK_IFF, TOK_COMMA, TOK_VAR};

TokType toktype;

void skip_spaces(){
    while(posi < input.size() && isspace((unsigned char)input[posi])) posi++;
}

bool isVarStart(char c){ return isupper((unsigned char)c); }
bool isVarChar(char c){ return isalpha((unsigned char)c) || isdigit((unsigned char)c); }

void next_token(){
    skip_spaces();
    if(posi >= input.size()){ toktype = TOK_END; token=""; return; }
    char c = input[posi];
    if(c == '('){ posi++; toktype = TOK_LP; token="("; return;}
    if(c == ')'){ posi++; toktype = TOK_RP; token=")"; return;}
    if(c == '!'){ posi++; toktype = TOK_NOT; token="!"; return;}
    if(c == '+'){ posi++; toktype = TOK_AND; token="+"; return;}
    if(c == '-'){ posi++; toktype = TOK_OR; token="-"; return;}
    if(c == '>'){ posi++; toktype = TOK_IMP; token=">"; return;}
    if(c == '='){ posi++; toktype = TOK_IFF; token="="; return;}
    if(c == ','){ posi++; toktype = TOK_COMMA; token=","; return;}
    if(isVarStart(c)){
        size_t s = posi;
        posi++;
        while(posi < input.size() && isVarChar(input[posi])) posi++;
        token = input.substr(s, posi - s);
        toktype = TOK_VAR;
        return;
    }
    // unknown char -> treat as error by consuming one char
    token = string(1, c);
    posi++;
    toktype = TOK_END;
}

bool syntax_error_flag = false;
string syntax_error_msg;

shared_ptr<Node> parse_expression();

// parse primary: VAR | !primary | (expr)
shared_ptr<Node> parse_primary(){
    if(toktype == TOK_END){ syntax_error_flag = true; syntax_error_msg = "unexpected end"; return nullptr; }
    if(toktype == TOK_VAR){
        auto n = Node::Var(token);
        next_token();
        return n;
    }
    if(toktype == TOK_NOT){
        next_token();
        auto a = parse_primary();
        if(!a) return nullptr;
        return Node::Not(a);
    }
    if(toktype == TOK_LP){
        next_token();
        auto a = parse_expression();
        if(!a) return nullptr;
        if(toktype != TOK_RP){ syntax_error_flag = true; syntax_error_msg = "missing )"; return nullptr; }
        next_token();
        return a;
    }
    syntax_error_flag = true;
    syntax_error_msg = "invalid token: " + token;
    return nullptr;
}

// parse binary operators with precedence:
// lowest: IFF (=)
// then IMP (>)
// then OR (-)
// then AND (+)
// left-associative for these
shared_ptr<Node> parse_binary(function<shared_ptr<Node>(shared_ptr<Node>, shared_ptr<Node>)> make_node, 
                              const TokType op, function<shared_ptr<Node>()> subparser)
{
    auto left = subparser();
    if(!left) return nullptr;
    while(op == toktype){
        next_token();
        auto right = subparser();
        if(!right) return nullptr;

        left = make_node(left, right);
    }
    return left;
}

shared_ptr<Node> parse_and(){
    return parse_binary(Node::And, TOK_AND, parse_primary);
}
shared_ptr<Node> parse_or(){
    // parse_and handles AND *FIRST*
    return parse_binary(Node::Or, TOK_OR, parse_and);
}
shared_ptr<Node> parse_imp(){
    // parse_or handles OR, AND *FIRST*
    return parse_binary(Node::Imp, TOK_IMP, parse_or);
}
shared_ptr<Node> parse_iff(){
    // parse_imp handles IMP, OR, AND *FIRST*
    return parse_binary(Node::Iff, TOK_IFF, parse_imp);
}

shared_ptr<Node> parse_expression(){
    return parse_iff();
}

// Utilities: convert to NNF (negation normal form), eliminate IMP and IFF, then CNF distribute OR over AND
shared_ptr<Node> eliminate_imp_iff(shared_ptr<Node> n){
    if(!n) return n;
    if(n->t == VAR) return n;
    if(n->t == NOT){
        auto c = eliminate_imp_iff(n->l);
        return Node::Not(c);
    }
    if(n->t == IMP){
        // A > B  ===  !A OR B
        auto A = eliminate_imp_iff(n->l);
        auto B = eliminate_imp_iff(n->r);
        return Node::Or(Node::Not(A), B);
    }
    if(n->t == IFF){
        // A = B === (A > B) AND (B > A)
        auto A = eliminate_imp_iff(n->l);
        auto B = eliminate_imp_iff(n->r);
        return Node::And(Node::Or(Node::Not(A), B), Node::Or(Node::Not(B), A));
    }
    if(n->t == AND || n->t == OR){
        auto L = eliminate_imp_iff(n->l);
        auto R = eliminate_imp_iff(n->r);
        if(n->t == AND) return Node::And(L,R);
        else return Node::Or(L,R);
    }
    return n;
}

shared_ptr<Node> toNNF(shared_ptr<Node> n){
    // shouldn't see IMP/IFF because eliminated!
    if(n->t == NOT){
        auto c = n->l;
        if(c->t == VAR) return n;
        if(c->t == NOT) return toNNF(c->l); // !!A -> A
        if(c->t == AND){
            // !(A & B) -> !A | !B
            return toNNF(Node::Or(Node::Not(c->l), Node::Not(c->r)));
        }
        if(c->t == OR){
            // !(A | B) -> !A & !B
            return toNNF(Node::And(Node::Not(c->l), Node::Not(c->r)));
        }
    } else if(n->t == AND || n->t == OR){
        return (n->t == AND) ? Node::And(toNNF(n->l), toNNF(n->r)) : Node::Or(toNNF(n->l), toNNF(n->r));
    }
    // else if(n->t == VAR) 
    return n;
}

// Clause representation: literal is (string name, bool isNeg)
// Clause is set of literals (no duplicates). We'll use sorted vector for deterministic ordering.
struct Lit {
    string var;
    bool neg;
    bool operator<(Lit const& o) const {
        if(var != o.var) return var < o.var;
        return neg < o.neg;
    }
    bool operator==(Lit const& o) const { return var==o.var && neg==o.neg;}
    string toString() const { return (neg ? "!" : "") + var; }
};

// Literal complement check
bool complementary(const Lit &a, const Lit &b){
    return a.var == b.var && a.neg != b.neg;
}

using Clause = vector<Lit>;

void normalize_clause(Clause& c) {
    sort(c.begin(), c.end());
    c.erase(unique(c.begin(), c.end()), c.end());
}

void add_clause_unique(vector<Clause>& clauses, Clause c, bool skip_dupCheck = false){
    normalize_clause(c);
    // This is **optional**: tautology elimination: if contains A and !A, skip adding clause (tautology)
    // for(size_t i=1;i<c.size();++i){
    //     if(complementary(c[i], c[i-1])) 
    //         return;
    // }
    // check duplicate
    if (!skip_dupCheck) {
        for(auto &ex : clauses) if(ex == c) return;
    }
    clauses.push_back(c);
}

// Convert NNF node into CNF clauses (distribute Or over And)
// Returns an expression in form of either conjunction of clauses; implement recursively:
// - If node is AND: CNF(node) = CNF(left) U CNF(right)
// - If node is OR: need to distribute: CNF(left OR right) = pairwise distribute clause sets.
vector<Clause> nodeToCNF(shared_ptr<Node> n){
    if(!n) return {};
    if(n->t == VAR){
        Clause c; c.push_back({n->name, false});
        return vector<Clause>{c};
    }
    if(n->t == NOT){
        // n->l must be VAR in NNF
        Clause c; c.push_back({n->l->name, true});
        return vector<Clause>{c};
    }
    if(n->t == AND){
        auto L = nodeToCNF(n->l);
        auto R = nodeToCNF(n->r);
        // conjunction: just merge clause lists
        vector<Clause> out = L;
        for(auto &c: R) out.push_back(c);
        return out;
    }
    if(n->t == OR){
        auto L = nodeToCNF(n->l);
        auto R = nodeToCNF(n->r);
        // distribution: for each clause a in L and b in R, produce merged clause a U b
        vector<Clause> out;
        for(auto &a: L){
            for(auto &b: R){
                Clause m = a;
                m.insert(m.end(), b.begin(), b.end()); // a :: b
                add_clause_unique(out, m, true); // skip duplicate check here for efficiency
            }
        }
        
        sort(out.begin(), out.end(), [](Clause const& a, Clause const& b){
            if(a.size()!=b.size()) return a.size()<b.size();
            for(size_t i=0;i<a.size();++i) if(a[i].var!=b[i].var) return a[i].var<b[i].var;
            for(size_t i=0;i<a.size();++i) if(a[i].neg!=b[i].neg) return a[i].neg < b[i].neg;
            return false;
        });
        out.erase(unique(out.begin(), out.end()), out.end()); // remove duplicates
        return out;
    }
    return {};
}

// Helper: parse multiple sentences separated by commas into AST list
vector<shared_ptr<Node>> parse_sentences(const string &s){
    input = s;
    posi = 0;
    next_token();
    vector<shared_ptr<Node>> res;
    while(toktype != TOK_END){
        auto node = parse_expression();
        if(!node) return {}; // parse error flagged
        res.push_back(node);
        if(toktype == TOK_COMMA){ next_token(); continue; }
        else if(toktype == TOK_END) break;
        else {
            // extra garbage
            syntax_error_flag = true;
            syntax_error_msg = "unexpected token after expression: " + token;
            return {};
        }
    }
    return res;
}


string clause_to_string(const Clause &c){
    if(c.empty()) return "0"; // empty clause
    if(c.size() == 1) 
        return c[0].toString();
    string s="(";
    for(size_t i=0;i<c.size();++i){
        s += c[i].toString();
        if(i+1<c.size()) s += "-";
    }
    s += ")";
    return s;
}

// Format CNF (vector<Clause>) to string in the required output style: clauses joined by +,
// each clause: if multiple literals, parenthesize and join by '-', single literal printed as is.
string CNF_to_string(const vector<Clause>& cnf){
    string out;
    for(size_t i=0;i<cnf.size();++i){
        auto &c = cnf[i];
        out += clause_to_string(c);
        if(i+1 < cnf.size()) out += "+";
    }
    return out;
}


// Resolve two clauses c1 and c2, return vector of resolvents (could be multiple)
vector<Clause> Resolve(const Clause &c1, const Clause &c2) {
    vector<Clause> resolvents;
    for(size_t i = 0; i < c1.size(); ++i) {
        for(size_t j = 0; j < c2.size(); ++j) {
            if(complementary(c1[i], c2[j])) {
                Clause resolvent;
                // add all literals from c1 except c1[i]
                resolvent.insert(resolvent.end(), c1.begin(), c1.begin() + i);
                resolvent.insert(resolvent.end(), c1.begin() + i + 1, c1.end());
                // add all literals from c2 except c2[j]
                resolvent.insert(resolvent.end(), c2.begin(), c2.begin() + j);
                resolvent.insert(resolvent.end(), c2.begin() + j + 1, c2.end());
                // normalize: sort and unique
                normalize_clause(resolvent);
                resolvents.push_back(resolvent);
            }
        }
    }
    return resolvents;
}





// Resolution prover using BFS to find minimal derivation for empty clause
// We'll maintain a list `allClauses` (initial + derived). Each clause has an index.
// Use queue to explore newly added clauses; for each new clause, try resolve with older ones.
// Track parents: parent pair (i,j) for each derived clause (i<j).
// Stop when empty clause found; then backtrack to output the sequence of derived clauses needed.
//
// For output sequence we will produce:
// - first line: the initial CNF (KB U ¬alpha) as a single string (clauses joined by '+')
// - then the derived clauses in an order that leads to empty clause (a minimal set). For simplicity,
//   we reconstruct the derivation tree and output derived clauses in BFS order from initial leaves to root.
//   The sample expects a small sequence; our reconstruction will be reasonable.
struct Parent { int a,b; };
struct Deriv {
    Clause clause;
    bool is_initial;
    Parent parent; // valid if not initial
};

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string kb_line;
    string alpha_line;
    if(!getline(cin, kb_line)){ cerr<<"need two lines of input\n"; return 1;}
    if(!getline(cin, alpha_line)){ cerr<<"need second line (conclusion)\n"; return 1;}

    // parse KB sentences
    auto kb_nodes = parse_sentences(kb_line);
    if(syntax_error_flag){ cout<<"syntax error\n"; return 0; }

    // parse alpha
    auto tmp = parse_sentences(alpha_line);
    if(tmp.size() != 1){ cout<<"syntax error\n"; return 0; }
    auto alpha = tmp[0];
    if(syntax_error_flag){ cout<<"syntax error\n"; return 0; }

    // Convert each KB sentence: eliminate IMP/IFF -> toNNF -> CNF clauses
    vector<Clause> initClauses;
    for(auto &node : kb_nodes){
        auto eliminated = eliminate_imp_iff(node);
        auto nnf = toNNF(eliminated);
        auto cnf = nodeToCNF(nnf);
        for(auto &c: cnf) add_clause_unique(initClauses, c);
    }
    // add negation of alpha
    auto not_alpha = Node::Not(alpha);
    auto elim2 = eliminate_imp_iff(not_alpha);
    auto nnf2 = toNNF(elim2);
    auto cnf2 = nodeToCNF(nnf2);
    for(auto &c: cnf2) add_clause_unique(initClauses, c);

    // initial CNF string
    string initialCNFStr = CNF_to_string(initClauses);
    // cout << "KB ⊢ α: " << initialCNFStr << endl;

    // prepare resolution structures
    vector<Deriv> all;
    for(auto &c: initClauses){
        Deriv d; d.clause = c; d.is_initial = true; d.parent = {-1,-1};
        all.push_back(d);
    }

    // queue of pairs to try? We'll BFS by layers: for each newly added clause, try resolve with all previous clauses.
    int head = 0;
    bool found = false;
    int empty_index = -1;

    // Use unordered_map<string,int> to avoid duplicates: key as serialized clause
    unordered_map<string, int> seen;
    for(size_t i = 0; i < initClauses.size(); ++i) {
        seen[clause_to_string(initClauses[i])] = (int)i;
    }

    while(head < (int)all.size() && !found){
        // resolve clause at head with all earlier clauses (including earlier derived ones)
        for(int j = 0; j < head && !found; ++j){
            Clause &C1 = all[head].clause;
            Clause &C2 = all[j].clause;

            // try all pairs of literals one from C1 and one from C2 that are complementary
            auto resolvents = Resolve(C1, C2);
            for(auto &resolvent : resolvents){
                // check if we've seen it
                string key = clause_to_string(resolvent);
                if(seen.find(key) != seen.end()) continue;
                // new clause
                Deriv D; D.clause = resolvent; D.is_initial = false; D.parent = {j, head};
                // push
                int idx = all.size();
                all.push_back(D);
                seen[key] = idx;
                // check for empty clause
                if(resolvent.empty()){
                    found = true;
                    empty_index = idx;
                    break;
                }
            }
        }
        head++;
    }

    if(found){
        cout << "yes\n";
        // reconstruct minimal derivation: backtrack from empty_index, collect indices used
        // We'll collect in reverse order via DFS on parents to get a vector of indices required.
        vector<pair<int, Parent>> used;
        function<void(int)> collect = [&](int idx){
            if(all[idx].is_initial) return;
            used.push_back({idx, all[idx].parent});
            collect(all[idx].parent.a);
            collect(all[idx].parent.b);
        };
        collect(empty_index);
        reverse(used.begin(), used.end()); // now from initial to empty

        int step = 0;
        vector<bool> needed(all.size(), false);
        for (int i = 0; i < (int)initClauses.size(); ++i) {
            needed[i] = true; // initial clauses are needed
        }

        cout << (int)used.size() + 1 << "\n"; // number of derived clauses, including initial CNF
        cout << CNF_to_string(initClauses) << "\n";

        while (step < (int)used.size()) {
            int idx = used[step].first;
            Parent p = used[step].second;
            needed[idx] = true;  // add derived
            needed[p.a] = false;
            needed[p.b] = false; // remove parents
            string s;
            for (int i = 0; i < (int)all.size(); ++i) {
                if (needed[i]) {
                    if(!s.empty()) s += "+";
                    s += clause_to_string(all[i].clause);
                }
            }
            cout << s << "\n";
            step++;
        }
            
    } else {
        cout << "no\n";
    }

    return 0;
}
