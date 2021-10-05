#ifndef LDLT_MACROS_HPP_TSAOHJEXS
#define LDLT_MACROS_HPP_TSAOHJEXS

#include "ldlt/detail/hedley.h"

#define LDLT_PP_CALL(M, ...) M(__VA_ARGS__)
#define LDLT_PP_CALL2(M, ...) M(__VA_ARGS__)

#define LDLT_IMPL_PP_CAT(A, ...) A##__VA_ARGS__
#define LDLT_IMPL_PP_CAT2(A, ...) A##__VA_ARGS__

#define LDLT_IMPL_PP_CONSUME(X)

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

#define LDLT_IMPL_PP_DEC_1 0
#define LDLT_IMPL_PP_DEC_2 1
#define LDLT_IMPL_PP_DEC_3 2
#define LDLT_IMPL_PP_DEC_4 3
#define LDLT_IMPL_PP_DEC_5 4
#define LDLT_IMPL_PP_DEC_6 5
#define LDLT_IMPL_PP_DEC_7 6
#define LDLT_IMPL_PP_DEC_8 7
#define LDLT_IMPL_PP_DEC_9 8
#define LDLT_IMPL_PP_DEC_10 9
#define LDLT_IMPL_PP_DEC_11 10
#define LDLT_IMPL_PP_DEC_12 11
#define LDLT_IMPL_PP_DEC_13 12
#define LDLT_IMPL_PP_DEC_14 13
#define LDLT_IMPL_PP_DEC_15 14
#define LDLT_IMPL_PP_DEC_16 15
#define LDLT_IMPL_PP_DEC_17 16
#define LDLT_IMPL_PP_DEC_18 17
#define LDLT_IMPL_PP_DEC_19 18
#define LDLT_IMPL_PP_DEC_20 19
#define LDLT_IMPL_PP_DEC_21 20
#define LDLT_IMPL_PP_DEC_22 21
#define LDLT_IMPL_PP_DEC_23 22
#define LDLT_IMPL_PP_DEC_24 23
#define LDLT_IMPL_PP_DEC_25 24
#define LDLT_IMPL_PP_DEC_26 25
#define LDLT_IMPL_PP_DEC_27 26
#define LDLT_IMPL_PP_DEC_28 27
#define LDLT_IMPL_PP_DEC_29 28
#define LDLT_IMPL_PP_DEC_30 29
#define LDLT_IMPL_PP_DEC_31 30
#define LDLT_IMPL_PP_DEC_32 31

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
#define LDLT_IMPL_PP_DEC_I(X) LDLT_IMPL_PP_DEC_##X
#define LDLT_IMPL_PP_DEC(X) LDLT_IMPL_PP_DEC_I(X)
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

#define LDLT_PP_HEAD(...) LDLT_IMPL_PP_HEAD_0(__VA_ARGS__)
#define LDLT_PP_TAIL(...) LDLT_IMPL_PP_TAIL_0(__VA_ARGS__)

#define LDLT_PP_2ND(...) LDLT_PP_HEAD(LDLT_PP_TAIL(__VA_ARGS__))
#define LDLT_PP_3RD(...) LDLT_PP_HEAD(LDLT_PP_TAIL(LDLT_PP_TAIL(__VA_ARGS__)))
#define LDLT_PP_4TH(...)                                                       \
	LDLT_PP_HEAD(LDLT_PP_TAIL(LDLT_PP_TAIL(LDLT_PP_TAIL(__VA_ARGS__))))

#define LDLT_PP_STRINGIZE(...) LDLT_IMPL_PP_STRINGIZE(__VA_ARGS__)
#define LDLT_PP_CAT(A, ...) LDLT_IMPL_PP_CAT(A, __VA_ARGS__)
#define LDLT_PP_CAT2(A, ...) LDLT_IMPL_PP_CAT2(A, __VA_ARGS__)

#define LDLT_ID(Name)                                                          \
	LDLT_PP_CAT(LDLT_PP_CAT(_ldlt_, LDLT_PP_CAT(Name, _)), __LINE__)

#define LDLT_PP_REMOVE_PAREN(...)                                              \
	LDLT_IMPL_PP_REMOVE_PAREN2(LDLT_IMPL_PP_REMOVE_PAREN1 __VA_ARGS__)
#define LDLT_PP_REMOVE_PAREN1(...)                                             \
	LDLT_IMPL_PP_REMOVE_PAREN21(LDLT_IMPL_PP_REMOVE_PAREN11 __VA_ARGS__)
#define LDLT_PP_UNWRAP(...) LDLT_PP_HEAD __VA_ARGS__ LDLT_PP_TAIL __VA_ARGS__

#define LDLT_REMOVE_PAREN(...) __VA_ARGS__

#define LDLT_NOM_SEMICOLON static_assert(true, ".")

#define LDLT_PP_IMPL_IS_1(_0, _1, _2, _3, _4, _5, _7, N, ...) N
#define LDLT_PP_IS_SINGLE_ARG(...)                                             \
	LDLT_PP_IMPL_IS_1(__VA_ARGS__, 0, 0, 0, 0, 0, 0, 1, 1)

#define LDLT_PP_HEAD_ROBUST_0(x, ...) x
#define LDLT_PP_HEAD_ROBUST_1(...) __VA_ARGS__

#define LDLT_PP_TAIL_ROBUST_0(x, ...) __VA_ARGS__
#define LDLT_PP_TAIL_ROBUST_1(...)

#define LDLT_PP_HEAD_ROBUST(...)                                               \
	LDLT_PP_CAT2(LDLT_PP_HEAD_ROBUST_, LDLT_PP_IS_SINGLE_ARG(__VA_ARGS__))       \
	(__VA_ARGS__)

#define LDLT_PP_TAIL_ROBUST(...)                                               \
	LDLT_PP_CAT2(LDLT_PP_TAIL_ROBUST_, LDLT_PP_IS_SINGLE_ARG(__VA_ARGS__))       \
	(__VA_ARGS__)

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
#define LDLT_FREEA(x) ((void)x)
#else
#define LDLT_HAS_ALLOCA 0
#endif

#endif

#define LDLT_CACHELINE_BYTES 64

#define LDLT_IMPL_TO_ARGS_DISPATCH_Vec(Dim) Dim
#define LDLT_IMPL_TO_ARGS_DISPATCH_Mat(Rows, Cols) Rows, Cols

#define LDLT_IMPL_TO_ARGS(Dim) LDLT_PP_CAT(LDLT_IMPL_TO_ARGS_DISPATCH_, Dim)

#define LDLT_IMPL_ALIGN(I, _, NameTagDimAlignType)                             \
	::ldlt::usize{(LDLT_PP_4TH NameTagDimAlignType)},

#define LDLT_IMPL_LOCAL_DIMS_VEC2(I, Dim, Align, Type)                         \
	::ldlt::usize const LDLT_ID(LDLT_PP_CAT(dim, I)) = ::ldlt::usize(Dim);       \
	::ldlt::usize const LDLT_ID(LDLT_PP_CAT(size_bytes, I)) =                    \
			::ldlt::detail::uround_up(                                               \
					sizeof(Type) * LDLT_ID(LDLT_PP_CAT(dim, I)),                         \
					::ldlt::usize{(Align)});

#define LDLT_IMPL_LOCAL_DIMS_MAT2(I, Rows, Cols, Align, Type)                  \
	::ldlt::usize const LDLT_ID(LDLT_PP_CAT(rows, I)) = ::ldlt::usize(Rows);     \
	::ldlt::usize const LDLT_ID(LDLT_PP_CAT(cols, I)) = ::ldlt::usize(Cols);     \
	::ldlt::usize const LDLT_ID(LDLT_PP_CAT(stride_bytes, I)) =                  \
			::ldlt::detail::uround_up(                                               \
					sizeof(Type) * LDLT_ID(LDLT_PP_CAT(rows, I)),                        \
					::ldlt::usize{(Align)});                                             \
	::ldlt::usize const LDLT_ID(LDLT_PP_CAT(size_bytes, I)) =                    \
			LDLT_ID(LDLT_PP_CAT(cols, I)) * LDLT_ID(LDLT_PP_CAT(stride_bytes, I));

#define LDLT_IMPL_LOCAL_DIMS_VEC(I, Name, Tag, Dim, Align, Type)               \
	LDLT_IMPL_LOCAL_DIMS_VEC2(I, LDLT_IMPL_TO_ARGS(Dim), Align, Type)

#define LDLT_IMPL_LOCAL_DIMS_MAT(I, Name, Tag, Dim, Align, Type)               \
	LDLT_PP_CALL2(                                                               \
			LDLT_IMPL_LOCAL_DIMS_MAT2, I, LDLT_IMPL_TO_ARGS(Dim), Align, Type)

#define LDLT_IMPL_LOCAL_DIMS_DISPATCH_Vec(Dim) LDLT_IMPL_LOCAL_DIMS_VEC
#define LDLT_IMPL_LOCAL_DIMS_DISPATCH_Mat(Rows, Cols) LDLT_IMPL_LOCAL_DIMS_MAT

#define LDLT_IMPL_MIN_ALIGN_DISPATCH_Vec(Dim) LDLT_IMPL_MIN_ALIGN_VEC
#define LDLT_IMPL_MIN_ALIGN_DISPATCH_Mat(Rows, Cols) LDLT_IMPL_MIN_ALIGN_MAT

#define LDLT_IMPL_LOCAL_DIMS(I, _, NameTagDimAlignType)                        \
	LDLT_PP_CALL(                                                                \
			LDLT_PP_CAT(                                                             \
					LDLT_IMPL_LOCAL_DIMS_DISPATCH_, LDLT_PP_3RD NameTagDimAlignType),    \
			I,                                                                       \
			LDLT_PP_REMOVE_PAREN NameTagDimAlignType)                                \
	LDLT_ID(total_size_bytes) +=                                                 \
			LDLT_ID(LDLT_PP_CAT(size_bytes, I)) +                                    \
			(::ldlt::usize{LDLT_PP_4TH NameTagDimAlignType} - LDLT_ID(min_align));

#define LDLT_IMPL_MAKE_WORKSPACE2_Vec(I, Name, Align, Type)                    \
	::ldlt::VectorViewMut<Type>(Name) = {                                        \
			::ldlt::tags::FromPtrSize{},                                             \
			LDLT_ID(LDLT_PP_CAT(array_manager, I))._.data,                           \
			::ldlt::isize(LDLT_ID(LDLT_PP_CAT(dim, I))),                             \
	}

#define LDLT_IMPL_MAKE_WORKSPACE2_Mat(I, Name, Align, Type)                    \
	static_assert(                                                               \
			((sizeof(Type) % ::ldlt::usize{(Align)} == 0) ||                         \
	     (::ldlt::usize{(Align)} % sizeof(Type) == 0)),                          \
			".");                                                                    \
	::ldlt::MatrixViewMut<Type, ::ldlt::Layout::colmajor>(Name) = {              \
			::ldlt::tags::FromPtrRowsColsStride{},                                   \
			LDLT_ID(LDLT_PP_CAT(array_manager, I))._.data,                           \
			::ldlt::isize(LDLT_ID(LDLT_PP_CAT(rows, I))),                            \
			::ldlt::isize(LDLT_ID(LDLT_PP_CAT(cols, I))),                            \
			::ldlt::isize(LDLT_ID(LDLT_PP_CAT(stride_bytes, I)) / sizeof(Type)),     \
	}

#define LDLT_IMPL_MAKE_WORKSPACE2_DISPATCH_Vec(Dim)                            \
	LDLT_IMPL_MAKE_WORKSPACE2_Vec

#define LDLT_IMPL_MAKE_WORKSPACE2_DISPATCH_Mat(Rows, Cols)                     \
	LDLT_IMPL_MAKE_WORKSPACE2_Mat

#define LDLT_IMPL_MAKE_WORKSPACE2(I, Name, Tag, Dim, Align, Type)              \
	::ldlt::detail::ManagedArray<Type> LDLT_ID(LDLT_PP_CAT(array_manager, I)){   \
			::ldlt::detail::malloca_tags::Tag{},                                     \
			typename ::ldlt::detail::ManagedArray<Type>::Inner{                      \
					static_cast<Type*>(::ldlt::detail::next_aligned(                     \
							LDLT_ID(current_ptr), ::ldlt::usize{(Align)})),                  \
					(LDLT_ID(LDLT_PP_CAT(size_bytes, I)) / sizeof(Type)),                \
			},                                                                       \
	};                                                                           \
	LDLT_PP_CAT(LDLT_IMPL_MAKE_WORKSPACE2_DISPATCH_, Dim)(I, Name, Align, Type); \
	LDLT_ID(current_ptr) += LDLT_ID(LDLT_PP_CAT(size_bytes, I));

#define LDLT_IMPL_MAKE_WORKSPACE(I, _, NameTagDimAlignType)                    \
	LDLT_PP_CALL(                                                                \
			LDLT_IMPL_MAKE_WORKSPACE2, I, LDLT_PP_REMOVE_PAREN NameTagDimAlignType)

#define LDLT_WORKSPACE_MEMORY(Name, Tag, Dim, Align, Type)                     \
	LDLT_IMPL_MULTI_WORKSPACE_MEMORY(((Name, Tag, Dim, Align, Type)))

#define LDLT_MULTI_WORKSPACE_MEMORY(...)                                       \
	LDLT_IMPL_MULTI_WORKSPACE_MEMORY((__VA_ARGS__))

#define LDLT_IMPL_MULTI_WORKSPACE_MEMORY(NamesTagsDimsAlignsTypes)             \
	constexpr ::ldlt::usize LDLT_ID(min_align) =                                 \
			::ldlt::detail::min_list({LDLT_PP_TUPLE_FOR_EACH_I(                      \
					LDLT_IMPL_ALIGN, _, NamesTagsDimsAlignsTypes)});                     \
	::ldlt::usize LDLT_ID(total_size_bytes) = 0;                                 \
	LDLT_PP_TUPLE_FOR_EACH_I(LDLT_IMPL_LOCAL_DIMS, _, NamesTagsDimsAlignsTypes)  \
	LDLT_IMPL_GET_MALLOCA(                                                       \
			LDLT_ID(w_malloca), LDLT_ID(total_size_bytes), LDLT_ID(min_align));      \
	unsigned char* LDLT_ID(current_ptr) =                                        \
			static_cast<unsigned char*>(LDLT_ID(w_malloca)._.malloca_ptr);           \
	LDLT_PP_TUPLE_FOR_EACH_I(                                                    \
			LDLT_IMPL_MAKE_WORKSPACE, _, NamesTagsDimsAlignsTypes)                   \
	LDLT_NOM_SEMICOLON

#if LDLT_HAS_ALLOCA

#define LDLT_IMPL_GET_MALLOCA(Name, SizeBytes, Align)                          \
	static_assert(                                                               \
			::ldlt::usize(Align) > 0, "Align must be a constant expression");        \
	::ldlt::usize const LDLT_ID(malloca_size_bytes) =                            \
			::ldlt::usize(SizeBytes) + ::ldlt::usize{Align};                         \
	::ldlt::detail::ManagedMalloca const Name {                                  \
		::ldlt::detail::malloca_tags::Uninit{},                                    \
				::ldlt::detail::ManagedMalloca::Inner{                                 \
						((LDLT_ID(malloca_size_bytes)) <                                   \
		         ::ldlt::usize{LDLT_MAX_STACK_ALLOC_SIZE})                         \
								? (LDLT_ALLOCA(LDLT_ID(malloca_size_bytes)))                   \
								: (::std::malloc(LDLT_ID(malloca_size_bytes))),                \
						LDLT_ID(malloca_size_bytes),                                       \
				},                                                                     \
	}

#else

#define LDLT_IMPL_GET_MALLOCA(Name, SizeBytes, Align)                          \
	alignas(::ldlt::usize{Align}) unsigned char LDLT_ID(                         \
			buf)[LDLT_MAX_STACK_ALLOC_SIZE];                                         \
	::ldlt::usize const LDLT_ID(malloca_size_bytes) =                            \
			::ldlt::usize(SizeBytes) + ::ldlt::usize{Align};                         \
	::ldlt::detail::ManagedMalloca Name {                                        \
		::ldlt::detail::malloca_tags::Uninit{},                                    \
				::ldlt::detail::ManagedMalloca::Inner{                                 \
						(((SizeBytes)) < ::ldlt::usize{LDLT_MAX_STACK_ALLOC_SIZE})         \
								? static_cast<void*>(LDLT_ID(buf))                             \
								: (::std::malloc(LDLT_ID(malloca_size_bytes))),                \
						(SizeBytes),                                                       \
				},                                                                     \
	}

#endif

#if __cplusplus >= 201703L
#define LDLT_IF_CONSTEXPR if constexpr
#else
#define LDLT_IF_CONSTEXPR if
#endif

#define LDLT_TEMPLATE(TParams, Constraint, Attr_Name, ...)                     \
	LDLT_IMPL_TEMPLATE(                                                          \
			Attr_Name,                                                               \
			TParams,                                                                 \
			LDLT_PP_CAT2(LDLT_IMPL_PREFIX_, Constraint),                             \
			__VA_ARGS__)

#define LDLT_IMPL_PREFIX_requires
#define LDLT_IMPL_PREFIX_explicit

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define LDLT_HAS_CONCEPTS 1
#else
#define LDLT_HAS_CONCEPTS 0
#endif

#if LDLT_HAS_CONCEPTS
#define LDLT_IMPL_TEMPLATE(Attr_Name, TParams, Constraint, ...)                \
	template <LDLT_PP_REMOVE_PAREN(TParams)>                                     \
	requires Constraint Attr_Name LDLT_PP_TUPLE_TRANSFORM_I(                     \
			LDLT_IMPL_PARAM_EXPAND, _, (__VA_ARGS__))
#else

#define LDLT_IMPL_TEMPLATE2_HELPER_0(Constraint, Param)                        \
	LDLT_PP_TAIL Param LDLT_PP_HEAD Param

#define LDLT_IMPL_TEMPLATE2_HELPER_1(Constraint, Param)                        \
	::ldlt::detail::DiscardFirst<                                                \
			::ldlt::detail::enable_if_t<(Constraint), void>,                         \
			LDLT_PP_TAIL Param>                                                      \
			LDLT_PP_HEAD Param

#define LDLT_IMPL_TEMPLATE2_HELPER(I, Constraint, Param)                       \
	LDLT_PP_CAT2(                                                                \
			LDLT_IMPL_TEMPLATE2_HELPER_, LDLT_IMPL_PP_IS_1(LDLT_IMPL_PP_INC(I)))     \
	(Constraint, Param)

#define LDLT_IMPL_TEMPLATE(Attr_Name, TParams, Constraint, ...)                \
	template <LDLT_PP_REMOVE_PAREN(TParams)>                                     \
	Attr_Name LDLT_PP_TUPLE_TRANSFORM_I(                                         \
			LDLT_IMPL_TEMPLATE2_HELPER, Constraint, (__VA_ARGS__))

#endif

#if LDLT_HAS_CONCEPTS

#define LDLT_IMPL_AND(_, Param) &&LDLT_PP_UNWRAP(Param)
#define LDLT_IMPL_OR(_, Param) || LDLT_PP_UNWRAP(Param)

#define LDLT_IMPL_CONJUNCTION(First, ...)                                      \
	(LDLT_PP_UNWRAP(First)                                                       \
	     LDLT_PP_TUPLE_FOR_EACH(LDLT_IMPL_AND, _, (__VA_ARGS__)))

#define LDLT_IMPL_DISJUNCTION(First, ...)                                      \
	(LDLT_PP_UNWRAP(First) LDLT_PP_TUPLE_FOR_EACH(LDLT_IMPL_OR, _, (__VA_ARGS__)))

#define LDLT_DEF_CONCEPT(Tpl, Name, ...)                                       \
	template <LDLT_PP_REMOVE_PAREN1(Tpl)>                                        \
	concept Name = __VA_ARGS__

#define LDLT_CHECK_CONCEPT_MACRO(Namespace, ...)                               \
	static_assert(                                                               \
			Namespace::__VA_ARGS__, LDLT_PP_STRINGIZE(__VA_ARGS__) " failed")
#define LDLT_CONCEPT_MACRO(Namespace, ...) Namespace::__VA_ARGS__

#define LDLT_DEF_CONCEPT_CONJUNCTION(Tpl, Name, Terms)                         \
	LDLT_DEF_CONCEPT(Tpl, Name, LDLT_IMPL_CONJUNCTION Terms)
#define LDLT_DEF_CONCEPT_DISJUNCTION(Tpl, Name, Terms)                         \
	LDLT_DEF_CONCEPT(Tpl, Name, LDLT_IMPL_DISJUNCTION Terms)

#else

#if __cplusplus >= 201703L
#define LDLT_IMPL_DEF_CONCEPT(Tpl, Name, Value, ...)                           \
	namespace _ {                                                                \
	template <LDLT_PP_REMOVE_PAREN1(Tpl)>                                        \
	struct Name : __VA_ARGS__ {};                                                \
	}                                                                            \
	template <LDLT_PP_REMOVE_PAREN1(Tpl)>                                        \
	inline constexpr bool const& Name = Value
#elif __cplusplus >= 201402L
#define LDLT_IMPL_DEF_CONCEPT(Tpl, Name, Value, ...)                           \
	namespace _ {                                                                \
	template <LDLT_PP_REMOVE_PAREN1(Tpl)>                                        \
	struct Name : __VA_ARGS__ {};                                                \
	}                                                                            \
	namespace {                                                                  \
	template <LDLT_PP_REMOVE_PAREN1(Tpl)>                                        \
	constexpr bool const& Name = ::ldlt::detail::Constant<bool, Value>::value;   \
	}                                                                            \
	LDLT_NOM_SEMICOLON
#else
#define LDLT_IMPL_DEF_CONCEPT(Tpl, Name, Value, ...)                           \
	template <LDLT_PP_REMOVE_PAREN1(Tpl)>                                        \
	struct Name : __VA_ARGS__ {}
#endif

#if __cplusplus >= 201402L
#define LDLT_CONCEPT_MACRO(Namespace, ...) Namespace::__VA_ARGS__
#define LDLT_IMPL_ADD_VALUE(I, _, Param) (Param)
#define LDLT_IMPL_TRAIT(Param) LDLT_PP_HEAD Param _::LDLT_PP_TAIL Param
#else
#define LDLT_CONCEPT_MACRO(Namespace, ...) Namespace::__VA_ARGS__::value
#define LDLT_IMPL_ADD_VALUE(I, _, Param) ((LDLT_PP_REMOVE_PAREN(Param)::value))
#define LDLT_IMPL_TRAIT(Param) LDLT_PP_UNWRAP(Param)
#endif
#define LDLT_IMPL_PUT_TRAIT(I, _, Param) LDLT_IMPL_TRAIT(Param)

#define LDLT_CHECK_CONCEPT_MACRO(Namespace, ...)                               \
	static_assert(                                                               \
			decltype(Namespace::check_##__VA_ARGS__())::value,                       \
			LDLT_PP_STRINGIZE(__VA_ARGS__) " failed")
#define LDLT_DEF_CONCEPT(Tpl, Name, ...)                                       \
	LDLT_IMPL_DEF_CONCEPT(                                                       \
			Tpl, Name, (__VA_ARGS__), ::ldlt::detail::Constant<bool, __VA_ARGS__>);  \
	LDLT_TEMPLATE(                                                               \
			Tpl, requires(__VA_ARGS__), constexpr auto check_##Name, (_ = 0, int))   \
	noexcept->::ldlt::detail::Constant<bool, true>

#define LDLT_IMPL_SFINAE(_, Param)                                             \
	, ::ldlt::detail::enable_if_t<LDLT_PP_UNWRAP Param, int> = 0

#define LDLT_IMPL_OVERLOAD(Name_Tpl, Param)                                    \
	template <                                                                   \
			LDLT_PP_REMOVE_PAREN(LDLT_PP_TAIL Name_Tpl),                             \
			::ldlt::detail::enable_if_t<LDLT_PP_UNWRAP Param, int> = 0>              \
	auto LDLT_PP_CAT(                                                            \
			check_,                                                                  \
			LDLT_PP_HEAD Name_Tpl)() noexcept->::ldlt::detail::Constant<bool, true>;

#define LDLT_DEF_CONCEPT_BOOL_CONJUNCTION_IMPL(Tpl, Name, Base, Seq)           \
	LDLT_IMPL_DEF_CONCEPT(                                                       \
			Tpl,                                                                     \
			Name,                                                                    \
			(LDLT_PP_REMOVE_PAREN1(Base)::value),                                    \
			LDLT_PP_REMOVE_PAREN1(Base));                                            \
	template <LDLT_PP_REMOVE_PAREN(Tpl)                                          \
	              LDLT_PP_TUPLE_FOR_EACH(LDLT_IMPL_SFINAE, _, Seq)>              \
	auto check_##Name() noexcept->::ldlt::detail::Constant<bool, true>
#define LDLT_DEF_CONCEPT_BOOL_DISJUNCTION_IMPL(Tpl, Name, Base, Seq)           \
	LDLT_IMPL_DEF_CONCEPT(                                                       \
			Tpl,                                                                     \
			Name,                                                                    \
			(LDLT_PP_REMOVE_PAREN1(Base)::value),                                    \
			LDLT_PP_REMOVE_PAREN1(Base));                                            \
	LDLT_PP_TUPLE_FOR_EACH(LDLT_IMPL_OVERLOAD, (Name, Tpl), Seq)                 \
	LDLT_NOM_SEMICOLON

#define LDLT_DEF_CONCEPT_CONJUNCTION(Tpl, Name, Terms)                         \
	LDLT_DEF_CONCEPT_BOOL_CONJUNCTION_IMPL(                                      \
			Tpl,                                                                     \
			Name,                                                                    \
			(LDLT_IMPL_CONJUNCTION(Terms)),                                          \
			LDLT_PP_TUPLE_TRANSFORM_I(LDLT_IMPL_ADD_VALUE, _, Terms))

#define LDLT_DEF_CONCEPT_DISJUNCTION(Tpl, Name, Terms)                         \
	LDLT_DEF_CONCEPT_BOOL_DISJUNCTION_IMPL(                                      \
			Tpl,                                                                     \
			Name,                                                                    \
			(LDLT_IMPL_DISJUNCTION(Terms)),                                          \
			LDLT_PP_TUPLE_TRANSFORM_I(LDLT_IMPL_ADD_VALUE, _, Terms))

#define LDLT_IMPL_CONJUNCTION(Tuple)                                           \
	::ldlt::detail::Conjunction<LDLT_PP_REMOVE_PAREN(                            \
			LDLT_PP_TUPLE_TRANSFORM_I(LDLT_IMPL_PUT_TRAIT, _, Tuple))>
#define LDLT_IMPL_DISJUNCTION(Tuple)                                           \
	::ldlt::detail::Disjunction<LDLT_PP_REMOVE_PAREN(                            \
			LDLT_PP_TUPLE_TRANSFORM_I(LDLT_IMPL_PUT_TRAIT, _, Tuple))>

#endif

#define LDLT_CHECK_CONCEPT(...)                                                \
	LDLT_CHECK_CONCEPT_MACRO(::ldlt::concepts, __VA_ARGS__)
#define LDLT_CONCEPT(...) LDLT_CONCEPT_MACRO(::ldlt::concepts, __VA_ARGS__)

#ifdef __clang__
#define LDLT_MINSIZE __attribute__((minsize))
#else
#define LDLT_MINSIZE
#endif

#endif /* end of include guard LDLT_MACROS_HPP_TSAOHJEXS */
