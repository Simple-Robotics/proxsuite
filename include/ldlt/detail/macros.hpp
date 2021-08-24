#ifndef LDLT_MACROS_HPP_TSAOHJEXS
#define LDLT_MACROS_HPP_TSAOHJEXS

#include "ldlt/detail/hedley.h"

#define LDLT_IMPL_PP_CAT(A, ...) A##__VA_ARGS__

#define LDLT_IMPL_PP_CAT2(A, ...) A##__VA_ARGS__

#define LDLT_IMPL_PP_CONSUME(X)

#define LDLT_IMPL_PP_SEQ_HEAD_1(X)
#define LDLT_IMPL_PP_SEQ_HEAD_0(X) X LDLT_IMPL_PP_SEQ_HEAD_1(

#define LDLT_IMPL_PP_HEAD_0(arg, ...) arg
#define LDLT_IMPL_PP_TAIL_0(arg, ...) __VA_ARGS__
#define LDLT_IMPL_PP_HEAD_1(arg) arg
#define LDLT_IMPL_PP_TAIL_1(arg)

#define LDLT_IMPL_PP_STRINGIZE(...) #__VA_ARGS__

#define LDLT_IMPL_PP_REMOVE_PAREN1(...) LDLT_IMPL_PP_REMOVE_PAREN1 __VA_ARGS__
#define LDLT_IMPL_PP_REMOVE_PAREN2(...) LDLT_PP_CAT(LDLT_IMPL_PP, __VA_ARGS__)
#define LDLT_IMPL_PPLDLT_IMPL_PP_REMOVE_PAREN1

#define LDLT_IMPL_PP_REMOVE_PAREN11(...) LDLT_IMPL_PP_REMOVE_PAREN11 __VA_ARGS__
#define LDLT_IMPL_PP_REMOVE_PAREN21(...) LDLT_PP_CAT(LDLT_IMPL_PP, __VA_ARGS__)
#define LDLT_IMPL_PPLDLT_IMPL_PP_REMOVE_PAREN11

// clang-format off
#define LDLT_IMPL_PP_COUNT(Arg_0, Arg_1, Arg_2, Arg_3, Arg_4, Arg_5, Arg_6, Arg_7, Arg_8, Arg_9, Arg_10, Arg_11, Arg_12, Arg_13, Arg_14, Arg_15, Arg_16, Arg_17, Arg_18, Arg_19, Arg_20, Arg_21, Arg_22, Arg_23, Arg_24, Arg_25, Arg_26, Arg_27, Arg_28, Arg_29, Arg_30, Arg_31, N, ...) N

#define LDLT_IMPL_PP_IS_1_1 1
#define LDLT_IMPL_PP_IS_1_2 0
#define LDLT_IMPL_PP_IS_1_3 0
#define LDLT_IMPL_PP_IS_1_4 0
#define LDLT_IMPL_PP_IS_1_5 0
#define LDLT_IMPL_PP_IS_1_6 0
#define LDLT_IMPL_PP_IS_1_7 0
#define LDLT_IMPL_PP_IS_1_8 0
#define LDLT_IMPL_PP_IS_1_9 0
#define LDLT_IMPL_PP_IS_1_10 0
#define LDLT_IMPL_PP_IS_1_11 0
#define LDLT_IMPL_PP_IS_1_12 0
#define LDLT_IMPL_PP_IS_1_13 0
#define LDLT_IMPL_PP_IS_1_14 0
#define LDLT_IMPL_PP_IS_1_15 0
#define LDLT_IMPL_PP_IS_1_16 0
#define LDLT_IMPL_PP_IS_1_17 0
#define LDLT_IMPL_PP_IS_1_18 0
#define LDLT_IMPL_PP_IS_1_19 0
#define LDLT_IMPL_PP_IS_1_20 0
#define LDLT_IMPL_PP_IS_1_21 0
#define LDLT_IMPL_PP_IS_1_22 0
#define LDLT_IMPL_PP_IS_1_23 0
#define LDLT_IMPL_PP_IS_1_24 0
#define LDLT_IMPL_PP_IS_1_25 0
#define LDLT_IMPL_PP_IS_1_26 0
#define LDLT_IMPL_PP_IS_1_27 0
#define LDLT_IMPL_PP_IS_1_28 0
#define LDLT_IMPL_PP_IS_1_29 0
#define LDLT_IMPL_PP_IS_1_30 0
#define LDLT_IMPL_PP_IS_1_31 0
#define LDLT_IMPL_PP_IS_1_32 0
#define LDLT_IMPL_PP_IS_1_33 0

#define LDLT_IMPL_PP_INC_0 1
#define LDLT_IMPL_PP_INC_1 2
#define LDLT_IMPL_PP_INC_2 3
#define LDLT_IMPL_PP_INC_3 4
#define LDLT_IMPL_PP_INC_4 5
#define LDLT_IMPL_PP_INC_5 6
#define LDLT_IMPL_PP_INC_6 7
#define LDLT_IMPL_PP_INC_7 8
#define LDLT_IMPL_PP_INC_8 9
#define LDLT_IMPL_PP_INC_9 10
#define LDLT_IMPL_PP_INC_10 11
#define LDLT_IMPL_PP_INC_11 12
#define LDLT_IMPL_PP_INC_12 13
#define LDLT_IMPL_PP_INC_13 14
#define LDLT_IMPL_PP_INC_14 15
#define LDLT_IMPL_PP_INC_15 16
#define LDLT_IMPL_PP_INC_16 17
#define LDLT_IMPL_PP_INC_17 18
#define LDLT_IMPL_PP_INC_18 19
#define LDLT_IMPL_PP_INC_19 20
#define LDLT_IMPL_PP_INC_20 21
#define LDLT_IMPL_PP_INC_21 22
#define LDLT_IMPL_PP_INC_22 23
#define LDLT_IMPL_PP_INC_23 24
#define LDLT_IMPL_PP_INC_24 25
#define LDLT_IMPL_PP_INC_25 26
#define LDLT_IMPL_PP_INC_26 27
#define LDLT_IMPL_PP_INC_27 28
#define LDLT_IMPL_PP_INC_28 29
#define LDLT_IMPL_PP_INC_29 30
#define LDLT_IMPL_PP_INC_30 31
#define LDLT_IMPL_PP_INC_31 32
#define LDLT_IMPL_PP_INC_32 33

#define LDLT_IMPL_PP_INC_I(X) LDLT_IMPL_PP_INC_##X
#define LDLT_IMPL_PP_INC(X) LDLT_IMPL_PP_INC_I(X)
#define LDLT_IMPL_PP_IS_1_I(X) LDLT_IMPL_PP_IS_1_##X
#define LDLT_IMPL_PP_IS_1(X) LDLT_IMPL_PP_IS_1_I(X)

#define LDLT_IMPL_PP_COMMA_IF_NOT_0_1
#define LDLT_IMPL_PP_COMMA_IF_NOT_0_0 ,
#define LDLT_PP_COMMA_IF_NOT_0(X) LDLT_PP_CAT2(LDLT_IMPL_PP_COMMA_IF_NOT_0_, LDLT_PP_IS_0(X))

#define LDLT_PP_IS_1(X) LDLT_IMPL_PP_IS_1_I(X)
#define LDLT_PP_IS_0(X) LDLT_PP_IS_1(LDLT_IMPL_PP_INC_I(X))


#define LDLT_IMPL_PP_VARIADIC_SIZE(...) LDLT_IMPL_PP_COUNT(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define LDLT_IMPL_PP_MAKE_TUPLE0 ()
#define LDLT_IMPL_PP_MAKE_TUPLE1 (0)
#define LDLT_IMPL_PP_MAKE_TUPLE2 (0, 1)
#define LDLT_IMPL_PP_MAKE_TUPLE3 (0, 1, 2)
#define LDLT_IMPL_PP_MAKE_TUPLE4 (0, 1, 2, 3)
#define LDLT_IMPL_PP_MAKE_TUPLE5 (0, 1, 2, 3, 4)
#define LDLT_IMPL_PP_MAKE_TUPLE6 (0, 1, 2, 3, 4, 5)
#define LDLT_IMPL_PP_MAKE_TUPLE7 (0, 1, 2, 3, 4, 5, 6)
#define LDLT_IMPL_PP_MAKE_TUPLE8 (0, 1, 2, 3, 4, 5, 6, 7)
#define LDLT_IMPL_PP_MAKE_TUPLE9 (0, 1, 2, 3, 4, 5, 6, 7, 8)
#define LDLT_IMPL_PP_MAKE_TUPLE10 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
#define LDLT_IMPL_PP_MAKE_TUPLE11 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
#define LDLT_IMPL_PP_MAKE_TUPLE12 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
#define LDLT_IMPL_PP_MAKE_TUPLE13 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
#define LDLT_IMPL_PP_MAKE_TUPLE14 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
#define LDLT_IMPL_PP_MAKE_TUPLE15 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
#define LDLT_IMPL_PP_MAKE_TUPLE16 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
#define LDLT_IMPL_PP_MAKE_TUPLE17 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
#define LDLT_IMPL_PP_MAKE_TUPLE18 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
#define LDLT_IMPL_PP_MAKE_TUPLE19 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
#define LDLT_IMPL_PP_MAKE_TUPLE20 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
#define LDLT_IMPL_PP_MAKE_TUPLE21 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
#define LDLT_IMPL_PP_MAKE_TUPLE22 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
#define LDLT_IMPL_PP_MAKE_TUPLE23 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
#define LDLT_IMPL_PP_MAKE_TUPLE24 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
#define LDLT_IMPL_PP_MAKE_TUPLE25 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
#define LDLT_IMPL_PP_MAKE_TUPLE26 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
#define LDLT_IMPL_PP_MAKE_TUPLE27 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
#define LDLT_IMPL_PP_MAKE_TUPLE28 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
#define LDLT_IMPL_PP_MAKE_TUPLE29 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
#define LDLT_IMPL_PP_MAKE_TUPLE30 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
#define LDLT_IMPL_PP_MAKE_TUPLE31 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
#define LDLT_IMPL_PP_MAKE_TUPLE32 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
// clang-format on

#define LDLT_PP_MAKE_TUPLE(N) LDLT_PP_CAT2(LDLT_IMPL_PP_MAKE_TUPLE, N)

#define LDLT_IMPL_PP_TUPLE_FOR_EACH_1(Macro, Data, Tuple)                      \
	Macro(Data, LDLT_PP_REMOVE_PAREN1(Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_2(Macro, Data, Tuple)                      \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_1(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_3(Macro, Data, Tuple)                      \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_2(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_4(Macro, Data, Tuple)                      \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_3(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_5(Macro, Data, Tuple)                      \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_4(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_6(Macro, Data, Tuple)                      \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_5(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_7(Macro, Data, Tuple)                      \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_6(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_8(Macro, Data, Tuple)                      \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_7(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_9(Macro, Data, Tuple)                      \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_8(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_10(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_9(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_11(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_10(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_12(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_11(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_13(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_12(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_14(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_13(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_15(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_14(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_16(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_15(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_17(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_16(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_18(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_17(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_19(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_18(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_20(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_19(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_21(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_20(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_22(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_21(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_23(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_22(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_24(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_23(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_25(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_24(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_26(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_25(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_27(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_26(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_28(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_27(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_29(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_28(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_30(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_29(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_31(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_30(Macro, Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_32(Macro, Data, Tuple)                     \
	Macro(Data, LDLT_PP_HEAD Tuple)                                              \
			LDLT_IMPL_PP_TUPLE_FOR_EACH_31(Macro, Data, (LDLT_PP_TAIL Tuple))

#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_1(Macro, Start, Data, Tuple)             \
	Macro(Start, Data, LDLT_PP_REMOVE_PAREN1(Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_2(Macro, Start, Data, Tuple)             \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_1(      \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_3(Macro, Start, Data, Tuple)             \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_2(      \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_4(Macro, Start, Data, Tuple)             \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_3(      \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_5(Macro, Start, Data, Tuple)             \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_4(      \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_6(Macro, Start, Data, Tuple)             \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_5(      \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_7(Macro, Start, Data, Tuple)             \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_6(      \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_8(Macro, Start, Data, Tuple)             \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_7(      \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_9(Macro, Start, Data, Tuple)             \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_8(      \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_10(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_9(      \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_11(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_10(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_12(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_11(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_13(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_12(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_14(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_13(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_15(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_14(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_16(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_15(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_17(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_16(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_18(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_17(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_19(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_18(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_20(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_19(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_21(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_20(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_22(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_21(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_23(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_22(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_24(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_23(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_25(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_24(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_26(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_25(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_27(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_26(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_28(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_27(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_29(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_28(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_30(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_29(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_31(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_30(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))
#define LDLT_IMPL_PP_TUPLE_FOR_EACH_I_32(Macro, Start, Data, Tuple)            \
	Macro(Start, Data, LDLT_PP_HEAD Tuple) LDLT_IMPL_PP_TUPLE_FOR_EACH_I_31(     \
			Macro, LDLT_IMPL_PP_INC(Start), Data, (LDLT_PP_TAIL Tuple))

#define LDLT_IMPL_PP_TRANSFORM_HELPER(I, Macro_Data, Elem)                     \
	, LDLT_PP_HEAD Macro_Data(LDLT_IMPL_PP_INC(I), LDLT_PP_TAIL Macro_Data, Elem)

#define LDLT_IMPL_PP_TRANSFORM_0(Macro, Data, Tuple)                           \
	(Macro(0, Data, LDLT_PP_HEAD Tuple) LDLT_PP_TUPLE_FOR_EACH_I(                \
			LDLT_IMPL_PP_TRANSFORM_HELPER, (Macro, Data), (LDLT_PP_TAIL Tuple)))

#define LDLT_IMPL_PP_TRANSFORM_1(Macro, Data, Tuple)                           \
	(Macro(0, Data, LDLT_PP_REMOVE_PAREN1(Tuple)))

#define LDLT_PP_TUPLE_FOR_EACH_I(Macro, Data, Tuple)                           \
	LDLT_PP_CAT2(LDLT_IMPL_PP_TUPLE_FOR_EACH_I_, LDLT_PP_TUPLE_SIZE(Tuple))      \
	(Macro, 0, Data, Tuple)
#define LDLT_PP_TUPLE_FOR_EACH(Macro, Data, Tuple)                             \
	LDLT_PP_CAT2(LDLT_IMPL_PP_TUPLE_FOR_EACH_, LDLT_PP_TUPLE_SIZE(Tuple))        \
	(Macro, Data, Tuple)

#define LDLT_PP_TUPLE_TRANSFORM_I(Macro, Data, Tuple)                          \
	LDLT_PP_CAT2(                                                                \
			LDLT_IMPL_PP_TRANSFORM_, LDLT_IMPL_PP_IS_1(LDLT_PP_TUPLE_SIZE(Tuple)))   \
	(Macro, Data, Tuple)

#define LDLT_PP_TUPLE_SIZE(Tuple) LDLT_IMPL_PP_VARIADIC_SIZE Tuple
#define LDLT_PP_SEQ_HEAD(Seq)                                                  \
	LDLT_IMPL_PP_SEQ_HEAD_0 Seq)
#define LDLT_PP_SEQ_TAIL(Seq) LDLT_IMPL_PP_CONSUME Seq

#define LDLT_PP_HEAD(...) LDLT_IMPL_PP_HEAD_0(__VA_ARGS__)
#define LDLT_PP_TAIL(...) LDLT_IMPL_PP_TAIL_0(__VA_ARGS__)

#define LDLT_PP_STRINGIZE(...) LDLT_IMPL_PP_STRINGIZE(__VA_ARGS__)
#define LDLT_PP_CAT(A, ...) LDLT_IMPL_PP_CAT(A, __VA_ARGS__)
#define LDLT_PP_CAT2(A, ...) LDLT_IMPL_PP_CAT2(A, __VA_ARGS__)
#define LDLT_PP_REMOVE_PAREN(...)                                              \
	LDLT_IMPL_PP_REMOVE_PAREN2(LDLT_IMPL_PP_REMOVE_PAREN1 __VA_ARGS__)
#define LDLT_PP_REMOVE_PAREN1(...)                                             \
	LDLT_IMPL_PP_REMOVE_PAREN21(LDLT_IMPL_PP_REMOVE_PAREN11 __VA_ARGS__)
#define LDLT_PP_UNWRAP(...) LDLT_PP_HEAD __VA_ARGS__ LDLT_PP_TAIL __VA_ARGS__

#define LDLT_REMOVE_PAREN(...) __VA_ARGS__

#define LDLT_NOM_SEMICOLON static_assert(true, ".")

#ifndef LDLT_INLINE
#define LDLT_INLINE HEDLEY_ALWAYS_INLINE
#endif

#ifndef LDLT_NO_INLINE
#define LDLT_NO_INLINE HEDLEY_NEVER_INLINE
#endif

#define LDLT_FWD(x) (static_cast<decltype(x)&&>(x))
#define LDLT_DECLVAL(...) (static_cast<auto (*)()->__VA_ARGS__>(nullptr)())

#ifdef __clang__
#define LDLT_FP_PRAGMA _Pragma("STDC FP_CONTRACT ON")
#else
#define LDLT_FP_PRAGMA
#endif

#ifndef LDLT_MAX_STACK_ALLOC_SIZE
#define LDLT_MAX_STACK_ALLOC_SIZE (1024U * 8U) /* 8KiB */
#endif

#ifndef LDLT_HAS_ALLOCA

#if defined(HEDLEY_MSVC_VERSION)
#include <malloc.h>

#define LDLT_HAS_ALLOCA 1
#define LDLT_ALLOCA(x) (_malloca(x))
#define LDLT_FREEA(x) (_freea(x))
#elif defined(HEDLEY_GCC_VERSION) || defined(__clang__)
#include <alloca.h>

#define LDLT_HAS_ALLOCA 1
#define LDLT_ALLOCA(x) (alloca(x))
#define LDLT_FREEA(x) ((void)0)
#else
#define LDLT_HAS_ALLOCA 0
#endif

#endif

#if LDLT_HAS_ALLOCA

#define LDLT_WORKSPACE_MEMORY(Name, Count, Type)                               \
	::ldlt::usize const LDLT_PP_CAT(_ldlt_alloc, __LINE__) =                     \
			::ldlt::usize(Count) * sizeof(Type) +                                    \
			::ldlt::detail::UniqueMalloca<Type>::align;                              \
	::ldlt::detail::UniqueMalloca<Type> LDLT_PP_CAT(                             \
			_ldlt_malloca_handler, __LINE__){                                        \
			::ldlt::detail::UniqueMalloca<Type>::can_alloca(::ldlt::usize(Count))    \
					? (LDLT_ALLOCA(LDLT_PP_CAT(_ldlt_alloc, __LINE__)))                  \
					: ::std::malloc(LDLT_PP_CAT(_ldlt_alloc, __LINE__)),                 \
			::ldlt::usize(Count),                                                    \
	};                                                                           \
	Type* Name = LDLT_PP_CAT(_ldlt_malloca_handler, __LINE__).data;              \
	static_assert(true, ".")

#else

#define LDLT_WORKSPACE_MEMORY(Name, Count, Type)                               \
	alignas(::ldlt::detail::UniqueMalloca<Type>::align) unsigned char            \
			LDLT_PP_CAT(_ldlt_buf, __LINE__)[LDLT_MAX_STACK_ALLOC_SIZE];             \
	::ldlt::usize const LDLT_PP_CAT(_ldlt_alloc, __LINE__) =                     \
			::ldlt::usize(Count) * sizeof(Type) +                                    \
			::ldlt::detail::UniqueMalloca<Type>::align;                              \
	::ldlt::detail::UniqueMalloca<Type> LDLT_PP_CAT(                             \
			_ldlt_malloca_handler, __LINE__){                                        \
			::ldlt::detail::UniqueMalloca<Type>::can_alloca(::ldlt::usize(Count))    \
					? static_cast<void*>(LDLT_PP_CAT(_ldlt_buf, __LINE__))               \
					: ::std::malloc(LDLT_PP_CAT(_ldlt_alloc, __LINE__), )::ldlt::usize(  \
								Count),                                                        \
	};                                                                           \
	Type* Name = LDLT_PP_CAT(_ldlt_ldlt_malloca_handler, __LINE__).data;         \
	static_assert(true, ".")

#endif

#define LDLT_IMPL_PLUS(Type, Name_Count)                                       \
	+::ldlt::detail::round_up(                                                   \
			LDLT_PP_TAIL Name_Count,                                                 \
			::ldlt::detail::SimdCachelineAlignStep<Type>::value)

#define LDLT_IMPL_DECL_MEMORY(Type, Name_Count)                                \
	Type* LDLT_PP_HEAD Name_Count = LDLT_PP_CAT2(_ldlt_workspace, __LINE__);     \
	LDLT_PP_CAT2(_ldlt_workspace, __LINE__) += ::ldlt::detail::round_up(         \
			LDLT_PP_TAIL Name_Count,                                                 \
			::ldlt::detail::SimdCachelineAlignStep<Type>::value);

#define LDLT_MULTI_WORKSPACE_MEMORY(Names_Counts, Type)                        \
	::ldlt::i32 LDLT_PP_CAT2(_ldlt_alloc_size, __LINE__) =                       \
			LDLT_PP_TUPLE_FOR_EACH(LDLT_IMPL_PLUS, Type, Names_Counts);              \
	LDLT_WORKSPACE_MEMORY(                                                       \
			LDLT_PP_CAT2(_ldlt_workspace, __LINE__),                                 \
			LDLT_PP_CAT2(_ldlt_alloc_size, __LINE__),                                \
			Type);                                                                   \
	LDLT_PP_TUPLE_FOR_EACH(LDLT_IMPL_DECL_MEMORY, Type, Names_Counts)            \
	static_assert(true, ".")

#endif /* end of include guard LDLT_MACROS_HPP_TSAOHJEXS */
