#include "maxargs.h"

int maxargs(A_stm s)
{
    switch (s->kind)
    {
    case A_compoundStm:
        return max(maxargs(s->u.compound.stm1), maxargs(s->u.compound.stm2));
        break;
    case A_assignStm:
        return maxargs(s->u.assign.exp);
        break;
    case A_printStm:
        return max(countExpList(s->u.print.exps), maxargs(s->u.print.exps));
        break;
    }
}

int maxargs(A_exp e)
{
    switch (e->kind)
    {
    case A_idExp:
        return 0;
        break;
    case A_numExp:
        return 0;
        break;
    case A_opExp:
        return 0;
        break;
    case A_eseqExp:
        return maxargs(e->u.eseq.stm);
        break;
    }
}

int maxargs(A_expList el)
{
    switch (el->kind)
    {
    case A_pairExpList:
        return max(maxargs(el->u.pair.head), maxargs(el->u.pair.tail));
        break;
    case A_lastExpList:
        return maxargs(el->u.last);
        break;
    }
}

int countExpList(A_expList el)
{
    switch (el->kind)
    {
    case A_pairExpList:
        return 1 + countExpList(el->u.pair.tail);
        break;
    case A_lastExpList:
        return 1;
        break;
    }
}