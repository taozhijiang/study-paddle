# colorize CMake output

# code adapted from stackoverflow: http://stackoverflow.com/a/19578320
# from post authored by https://stackoverflow.com/users/2556117/fraser

macro(define_colors)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(ColourBold  "${Esc}[1m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
    set(BoldRed     "${Esc}[1;31m")
    set(BoldGreen   "${Esc}[1;32m")
    set(BoldYellow  "${Esc}[1;33m")
    set(BoldBlue    "${Esc}[1;34m")
    set(BoldMagenta "${Esc}[1;35m")
    set(BoldCyan    "${Esc}[1;36m")
    set(BoldWhite   "${Esc}[1;37m")
endmacro()

# 默认的GTest套件的链接库
set(GOOGLE_TEST_LIBS gtest gmock gtest_main)

macro(add_individual_test _TEST_NAME)
    set(extra_libs "${ARGN}")

    add_executable(${_TEST_NAME}_test
        ${_TEST_NAME}_test.cc)
    target_link_libraries(${_TEST_NAME}_test ${extra_libs} ${GOOGLE_TEST_LIBS} )
    add_test(NAME gtest_${_TEST_NAME}_test COMMAND ${_TEST_NAME}_test)
endmacro()
