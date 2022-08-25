#include "util.h"
#include "slp.h"

// table 表
// 将标识符映射到相应的整数值
typedef struct table *Table_;
struct table {string id; int value; Table_ tail};			// 表的数据结构
Table_ Table(string id, int value, struct table *tail) {	// 表的构造函数，空表为 NULL
    Table_ t = malloc(sizeof(*t));
    t->id=id; t->value=value; t->tail=tail;
    return t;
}

void interp(A_stm s) {
    
}

// interpStm 函数
// 函数输入为语句 s 和表 t，内部根据语句 s 的结果，在表中追加结点，最后输出新表。
Table_ interpStm(A_stm s, Table_ t) {
	
}


// interpExp 函数
// 表达式有返回值和副作用，因此设计数据结构 IntAndTable。
// 函数输入为表达式 e 和 表 t，内部完成表达式运算，修改表和返回值，输出 IntAndTable
struct IntAndTable {int i; Table_ t;};
struct IntAndTable interpExp(A_exp e, Table_ t) {
	
}