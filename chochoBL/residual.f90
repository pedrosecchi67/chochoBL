! for this extension: polynomials ought to be represented as two-column matrixes, with four lines
! for registration of polynomial coefficients
module residual
    use iso_c_binding

    real(8), parameter :: N1(1:16)=(/1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/), &
    N2(1:16)=(/0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/), &
    N3(1:16)=(/0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/), &
    N4(1:16)=(/0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/), &
    eps=1e-15, &
    log10eps=1.0000000000000022, &
    expit_lim=709.7
    integer, parameter :: Bp(1:16, 1:2)=reshape((/0, 0, 1, 0, 2, 0, 3, 0, 0, 1, 1, 1, 2, 1, 3, 1, &
    0, 2, 1, 2, 2, 2, 3, 2, 0, 3, 1, 3, 2, 3, 3, 3/), (/16, 2/), order=(/2, 1/))

contains

    subroutine getpoly(v, p)
        real(8), intent(IN) :: v(1:4)
        real(8), intent(OUT) :: p(1:16)

        p=v(1)*N1+v(2)*N2+v(3)*N3+v(4)*N4
        end subroutine getpoly

        subroutine getmult(p1, p2, m)
        real(8), intent(IN) :: p1(1:16), p2(1:16)
        real(8), intent(OUT) :: m(1:16)

        integer :: i, j, ni, nj

        m=0.0

        do i=1, 16
            do j=1, 16
                ni=Bp(i, 1)+Bp(j, 1)
                nj=Bp(i, 2)+Bp(j, 2)

                if((ni .lt. 4) .and. (nj .lt. 4)) then
                    m(4*nj+ni+1)=m(4*nj+ni+1)+p1(i)*p2(j)
                end if
            end do
        end do

    end subroutine getmult

    subroutine getint01(p, v)

        real(8), intent(IN) :: p(1:16)
        real(8), intent(OUT) :: v

        integer :: i

        v=0.0

        do i=1, 16
            ni=Bp(i, 1)
            nj=Bp(i, 2)

            v=v+p(4*nj+ni+1)/((ni+1)*(nj+1))
        end do

    end subroutine getint01

    subroutine getder_ksi(p, m)
        real(8), intent(IN) :: p(1:16)
        real(8), intent(OUT) :: m(1:16)

        integer :: n, ni, nj, nind

        m=0.0

        do n=1, 16
            ni=Bp(n, 1)-1
            nj=Bp(n, 2)

            if((ni .ge. 0) .and. (nj .ge. 0)) then
                nind=4*nj+ni
                m(nind+1)=m(nind+1)+p(n)*(ni+1)
            end if
        end do

    end subroutine getder_ksi

    subroutine getder_eta(p, m)
        real(8), intent(IN) :: p(1:16)
        real(8), intent(OUT) :: m(1:16)

        integer :: n, ni, nj, nind

        m=0.0

        do n=1, 16
            ni=Bp(n, 1)
            nj=Bp(n, 2)-1

            if((ni .ge. 0) .and. (nj .ge. 0)) then
                nind=4*nj+ni
                m(nind+1)=m(nind+1)+p(n)*(nj+1)
            end if
        end do

    end subroutine getder_eta

    subroutine getj(xs, ys, J)

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(OUT) :: J(1:16)

        real(8) :: px(1:16), py(1:16), p1(1:16), p2(1:16)

        call getpoly(xs, px)
        call getpoly(ys, py)

        call getder_ksi(px, p1)
        call getder_eta(py, p2)

        call getmult(p1, p2, J)

        call getder_ksi(py, p1)
        call getder_eta(px, p2)

        call getmult(p1, p2, px)

        J=J-px

    end subroutine getj

    subroutine surfint(xs, ys, poly, v)

        real(8), intent(IN) :: xs(1:4), ys(1:4), poly(1:16)
        real(8), intent(OUT) :: v

        real(8) :: J(1:16), val(1:16)

        call getJ(xs, ys, J)

        call getmult(poly, J, val)

        call getint01(val, v)

    end subroutine surfint

    subroutine dx_mult(xs, ys, p1, p2)

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(OUT) :: p1(1:16), p2(1:16)

        real(8) :: py(1:16)

        call getpoly(ys, py)

        call getder_eta(py, p1)
        call getder_ksi(-py, p2)

    end subroutine dx_mult

    subroutine surfint_dx(xs, ys, poly, v)

        real(8), intent(IN) :: xs(1:4), ys(1:4), poly(1:16)
        real(8), intent(OUT) :: v

        real(8) :: p1(1:16), p2(1:16), a(1:16), b(1:16), pvksi(1:16), pveta(1:16)

        call dx_mult(xs, ys, p1, p2)

        call getder_ksi(poly, pvksi)
        call getder_eta(poly, pveta)

        call getmult(pvksi, p1, a)
        call getmult(pveta, p2, b)

        call getint01(a+b, v)

    end subroutine surfint_dx

    subroutine dy_mult(xs, ys, p1, p2)

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(OUT) :: p1(1:16), p2(1:16)

        real(8) :: px(1:16)

        call getpoly(xs, px)

        call getder_eta(-px, p1)
        call getder_ksi(px, p2)

    end subroutine dy_mult

    subroutine surfint_dy(xs, ys, poly, v)

        real(8), intent(IN) :: xs(1:4), ys(1:4), poly(1:16)
        real(8), intent(OUT) :: v

        real(8) :: p1(1:16), p2(1:16), a(1:16), b(1:16), pvksi(1:16), pveta(1:16)

        call dy_mult(xs, ys, p1, p2)

        call getder_ksi(poly, pvksi)
        call getder_eta(poly, pveta)

        call getmult(pvksi, p1, a)
        call getmult(pveta, p2, b)

        call getint01(a+b, v)

    end subroutine surfint_dy

    subroutine surfint_udx(xs, ys, pu, pv, v)

        real(8), intent(IN) :: xs(1:4), ys(1:4), pu(1:16), pv(1:16)
        real(8), intent(OUT) :: v

        real(8) :: p1(1:16), p2(1:16), a(1:16), b(1:16), pvksi(1:16), pveta(1:16)

        call dx_mult(xs, ys, p1, p2)

        call getder_ksi(pv, pvksi)
        call getder_eta(pv, pveta)

        call getmult(pvksi, p1, a)
        call getmult(pveta, p2, b)

        call getmult(pu, a+b, p1)

        call getint01(p1, v)

    end subroutine surfint_udx

    subroutine surfint_udy(xs, ys, pu, pv, v)

        real(8), intent(IN) :: xs(1:4), ys(1:4), pu(1:16), pv(1:16)
        real(8), intent(OUT) :: v

        real(8) :: p1(1:16), p2(1:16), a(1:16), b(1:16), pvksi(1:16), pveta(1:16)

        call dy_mult(xs, ys, p1, p2)

        call getder_ksi(pv, pvksi)
        call getder_eta(pv, pveta)

        call getmult(pvksi, p1, a)
        call getmult(pveta, p2, b)

        call getmult(pu, a+b, p1)

        call getint01(p1, v)

    end subroutine surfint_udy

    subroutine get_rv_matrix(xs, ys, m)
    ! get matrix such that m(v_4) equals the residual of the function v according to Garlekin formulation

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(OUT) :: m(1:4, 1:4)

        real(8) :: poly(1:16)

        call getmult(N1, N1, poly)
        call surfint(xs, ys, poly, m(1, 1))
        call getmult(N1, N2, poly)
        call surfint(xs, ys, poly, m(1, 2))
        call getmult(N1, N3, poly)
        call surfint(xs, ys, poly, m(1, 3))
        call getmult(N1, N4, poly)
        call surfint(xs, ys, poly, m(1, 4))

        m(2, 1)=m(1, 2)
        call getmult(N2, N2, poly)
        call surfint(xs, ys, poly, m(2, 2))
        call getmult(N2, N3, poly)
        call surfint(xs, ys, poly, m(2, 3))
        call getmult(N2, N4, poly)
        call surfint(xs, ys, poly, m(2, 4))

        m(3, 1)=m(1, 3)
        m(3, 2)=m(2, 3)
        call getmult(N3, N3, poly)
        call surfint(xs, ys, poly, m(3, 3))
        call getmult(N3, N4, poly)
        call surfint(xs, ys, poly, m(3, 4))

        m(4, 1)=m(1, 4)
        m(4, 2)=m(2, 4)
        m(4, 3)=m(3, 4)
        call getmult(N4, N4, poly)
        call surfint(xs, ys, poly, m(4, 4))

    end subroutine get_rv_matrix

    subroutine get_rdvdx_matrix(xs, ys, m)
    ! get matrix such that m(v_4) equals the residual of the function dvdx according to Garlekin formulation

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(OUT) :: m(1:4, 1:4)

        call surfint_udx(xs, ys, N1, N1, m(1, 1))
        call surfint_udx(xs, ys, N1, N2, m(1, 2))
        call surfint_udx(xs, ys, N1, N3, m(1, 3))
        call surfint_udx(xs, ys, N1, N4, m(1, 4))

        call surfint_udx(xs, ys, N2, N1, m(2, 1))
        call surfint_udx(xs, ys, N2, N2, m(2, 2))
        call surfint_udx(xs, ys, N2, N3, m(2, 3))
        call surfint_udx(xs, ys, N2, N4, m(2, 4))

        call surfint_udx(xs, ys, N3, N1, m(3, 1))
        call surfint_udx(xs, ys, N3, N2, m(3, 2))
        call surfint_udx(xs, ys, N3, N3, m(3, 3))
        call surfint_udx(xs, ys, N3, N4, m(3, 4))

        call surfint_udx(xs, ys, N4, N1, m(4, 1))
        call surfint_udx(xs, ys, N4, N2, m(4, 2))
        call surfint_udx(xs, ys, N4, N3, m(4, 3))
        call surfint_udx(xs, ys, N4, N4, m(4, 4))

    end subroutine get_rdvdx_matrix

    subroutine get_rdvdy_matrix(xs, ys, m)
    ! get matrix such that m(v_4) equals the residual of the function dvdy according to Garlekin formulation

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(OUT) :: m(1:4, 1:4)

        call surfint_udy(xs, ys, N1, N1, m(1, 1))
        call surfint_udy(xs, ys, N1, N2, m(1, 2))
        call surfint_udy(xs, ys, N1, N3, m(1, 3))
        call surfint_udy(xs, ys, N1, N4, m(1, 4))

        call surfint_udy(xs, ys, N2, N1, m(2, 1))
        call surfint_udy(xs, ys, N2, N2, m(2, 2))
        call surfint_udy(xs, ys, N2, N3, m(2, 3))
        call surfint_udy(xs, ys, N2, N4, m(2, 4))

        call surfint_udy(xs, ys, N3, N1, m(3, 1))
        call surfint_udy(xs, ys, N3, N2, m(3, 2))
        call surfint_udy(xs, ys, N3, N3, m(3, 3))
        call surfint_udy(xs, ys, N3, N4, m(3, 4))

        call surfint_udy(xs, ys, N4, N1, m(4, 1))
        call surfint_udy(xs, ys, N4, N2, m(4, 2))
        call surfint_udy(xs, ys, N4, N3, m(4, 3))
        call surfint_udy(xs, ys, N4, N4, m(4, 4))

    end subroutine get_rdvdy_matrix

    subroutine intpqdrdx(xs, ys, p, q, r, v)

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(IN) :: p(1:16), q(1:16), r(1:16)
        real(8), intent(OUT) :: v

        real(8) :: poly(1:16)

        call getmult(p, q, poly)

        call surfint_udx(xs, ys, poly, r, v)

    end subroutine intpqdrdx

    subroutine intpqdrdy(xs, ys, p, q, r, v)

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(IN) :: p(1:16), q(1:16), r(1:16)
        real(8), intent(OUT) :: v

        real(8) :: poly(1:16)

        call getmult(p, q, poly)

        call surfint_udy(xs, ys, poly, r, v)

    end subroutine intpqdrdy

    subroutine get_rudvdx_matrix(xs, ys, m)

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(OUT) :: m(1:4, 1:4, 1:4)

        call intpqdrdx(xs, ys, N1, N1, N1, m(1, 1, 1))
        call intpqdrdx(xs, ys, N2, N1, N1, m(2, 1, 1))
        call intpqdrdx(xs, ys, N3, N1, N1, m(3, 1, 1))
        call intpqdrdx(xs, ys, N4, N1, N1, m(4, 1, 1))

        call intpqdrdx(xs, ys, N1, N2, N1, m(1, 2, 1))
        call intpqdrdx(xs, ys, N2, N2, N1, m(2, 2, 1))
        call intpqdrdx(xs, ys, N3, N2, N1, m(3, 2, 1))
        call intpqdrdx(xs, ys, N4, N2, N1, m(4, 2, 1))

        call intpqdrdx(xs, ys, N1, N3, N1, m(1, 3, 1))
        call intpqdrdx(xs, ys, N2, N3, N1, m(2, 3, 1))
        call intpqdrdx(xs, ys, N3, N3, N1, m(3, 3, 1))
        call intpqdrdx(xs, ys, N4, N3, N1, m(4, 3, 1))

        call intpqdrdx(xs, ys, N1, N4, N1, m(1, 4, 1))
        call intpqdrdx(xs, ys, N2, N4, N1, m(2, 4, 1))
        call intpqdrdx(xs, ys, N3, N4, N1, m(3, 4, 1))
        call intpqdrdx(xs, ys, N4, N4, N1, m(4, 4, 1))


        call intpqdrdx(xs, ys, N1, N1, N2, m(1, 1, 2))
        call intpqdrdx(xs, ys, N2, N1, N2, m(2, 1, 2))
        call intpqdrdx(xs, ys, N3, N1, N2, m(3, 1, 2))
        call intpqdrdx(xs, ys, N4, N1, N2, m(4, 1, 2))

        call intpqdrdx(xs, ys, N1, N2, N2, m(1, 2, 2))
        call intpqdrdx(xs, ys, N2, N2, N2, m(2, 2, 2))
        call intpqdrdx(xs, ys, N3, N2, N2, m(3, 2, 2))
        call intpqdrdx(xs, ys, N4, N2, N2, m(4, 2, 2))

        call intpqdrdx(xs, ys, N1, N3, N2, m(1, 3, 2))
        call intpqdrdx(xs, ys, N2, N3, N2, m(2, 3, 2))
        call intpqdrdx(xs, ys, N3, N3, N2, m(3, 3, 2))
        call intpqdrdx(xs, ys, N4, N3, N2, m(4, 3, 2))

        call intpqdrdx(xs, ys, N1, N4, N2, m(1, 4, 2))
        call intpqdrdx(xs, ys, N2, N4, N2, m(2, 4, 2))
        call intpqdrdx(xs, ys, N3, N4, N2, m(3, 4, 2))
        call intpqdrdx(xs, ys, N4, N4, N2, m(4, 4, 2))


        call intpqdrdx(xs, ys, N1, N1, N3, m(1, 1, 3))
        call intpqdrdx(xs, ys, N2, N1, N3, m(2, 1, 3))
        call intpqdrdx(xs, ys, N3, N1, N3, m(3, 1, 3))
        call intpqdrdx(xs, ys, N4, N1, N3, m(4, 1, 3))

        call intpqdrdx(xs, ys, N1, N2, N3, m(1, 2, 3))
        call intpqdrdx(xs, ys, N2, N2, N3, m(2, 2, 3))
        call intpqdrdx(xs, ys, N3, N2, N3, m(3, 2, 3))
        call intpqdrdx(xs, ys, N4, N2, N3, m(4, 2, 3))

        call intpqdrdx(xs, ys, N1, N3, N3, m(1, 3, 3))
        call intpqdrdx(xs, ys, N2, N3, N3, m(2, 3, 3))
        call intpqdrdx(xs, ys, N3, N3, N3, m(3, 3, 3))
        call intpqdrdx(xs, ys, N4, N3, N3, m(4, 3, 3))

        call intpqdrdx(xs, ys, N1, N4, N3, m(1, 4, 3))
        call intpqdrdx(xs, ys, N2, N4, N3, m(2, 4, 3))
        call intpqdrdx(xs, ys, N3, N4, N3, m(3, 4, 3))
        call intpqdrdx(xs, ys, N4, N4, N3, m(4, 4, 3))


        call intpqdrdx(xs, ys, N1, N1, N4, m(1, 1, 4))
        call intpqdrdx(xs, ys, N2, N1, N4, m(2, 1, 4))
        call intpqdrdx(xs, ys, N3, N1, N4, m(3, 1, 4))
        call intpqdrdx(xs, ys, N4, N1, N4, m(4, 1, 4))

        call intpqdrdx(xs, ys, N1, N2, N4, m(1, 2, 4))
        call intpqdrdx(xs, ys, N2, N2, N4, m(2, 2, 4))
        call intpqdrdx(xs, ys, N3, N2, N4, m(3, 2, 4))
        call intpqdrdx(xs, ys, N4, N2, N4, m(4, 2, 4))

        call intpqdrdx(xs, ys, N1, N3, N4, m(1, 3, 4))
        call intpqdrdx(xs, ys, N2, N3, N4, m(2, 3, 4))
        call intpqdrdx(xs, ys, N3, N3, N4, m(3, 3, 4))
        call intpqdrdx(xs, ys, N4, N3, N4, m(4, 3, 4))

        call intpqdrdx(xs, ys, N1, N4, N4, m(1, 4, 4))
        call intpqdrdx(xs, ys, N2, N4, N4, m(2, 4, 4))
        call intpqdrdx(xs, ys, N3, N4, N4, m(3, 4, 4))
        call intpqdrdx(xs, ys, N4, N4, N4, m(4, 4, 4))

    end subroutine get_rudvdx_matrix

    subroutine get_rudvdy_matrix(xs, ys, m)

        real(8), intent(IN) :: xs(1:4), ys(1:4)
        real(8), intent(OUT) :: m(1:4, 1:4, 1:4)

        call intpqdrdy(xs, ys, N1, N1, N1, m(1, 1, 1))
        call intpqdrdy(xs, ys, N2, N1, N1, m(2, 1, 1))
        call intpqdrdy(xs, ys, N3, N1, N1, m(3, 1, 1))
        call intpqdrdy(xs, ys, N4, N1, N1, m(4, 1, 1))

        call intpqdrdy(xs, ys, N1, N2, N1, m(1, 2, 1))
        call intpqdrdy(xs, ys, N2, N2, N1, m(2, 2, 1))
        call intpqdrdy(xs, ys, N3, N2, N1, m(3, 2, 1))
        call intpqdrdy(xs, ys, N4, N2, N1, m(4, 2, 1))

        call intpqdrdy(xs, ys, N1, N3, N1, m(1, 3, 1))
        call intpqdrdy(xs, ys, N2, N3, N1, m(2, 3, 1))
        call intpqdrdy(xs, ys, N3, N3, N1, m(3, 3, 1))
        call intpqdrdy(xs, ys, N4, N3, N1, m(4, 3, 1))

        call intpqdrdy(xs, ys, N1, N4, N1, m(1, 4, 1))
        call intpqdrdy(xs, ys, N2, N4, N1, m(2, 4, 1))
        call intpqdrdy(xs, ys, N3, N4, N1, m(3, 4, 1))
        call intpqdrdy(xs, ys, N4, N4, N1, m(4, 4, 1))


        call intpqdrdy(xs, ys, N1, N1, N2, m(1, 1, 2))
        call intpqdrdy(xs, ys, N2, N1, N2, m(2, 1, 2))
        call intpqdrdy(xs, ys, N3, N1, N2, m(3, 1, 2))
        call intpqdrdy(xs, ys, N4, N1, N2, m(4, 1, 2))

        call intpqdrdy(xs, ys, N1, N2, N2, m(1, 2, 2))
        call intpqdrdy(xs, ys, N2, N2, N2, m(2, 2, 2))
        call intpqdrdy(xs, ys, N3, N2, N2, m(3, 2, 2))
        call intpqdrdy(xs, ys, N4, N2, N2, m(4, 2, 2))

        call intpqdrdy(xs, ys, N1, N3, N2, m(1, 3, 2))
        call intpqdrdy(xs, ys, N2, N3, N2, m(2, 3, 2))
        call intpqdrdy(xs, ys, N3, N3, N2, m(3, 3, 2))
        call intpqdrdy(xs, ys, N4, N3, N2, m(4, 3, 2))

        call intpqdrdy(xs, ys, N1, N4, N2, m(1, 4, 2))
        call intpqdrdy(xs, ys, N2, N4, N2, m(2, 4, 2))
        call intpqdrdy(xs, ys, N3, N4, N2, m(3, 4, 2))
        call intpqdrdy(xs, ys, N4, N4, N2, m(4, 4, 2))


        call intpqdrdy(xs, ys, N1, N1, N3, m(1, 1, 3))
        call intpqdrdy(xs, ys, N2, N1, N3, m(2, 1, 3))
        call intpqdrdy(xs, ys, N3, N1, N3, m(3, 1, 3))
        call intpqdrdy(xs, ys, N4, N1, N3, m(4, 1, 3))

        call intpqdrdy(xs, ys, N1, N2, N3, m(1, 2, 3))
        call intpqdrdy(xs, ys, N2, N2, N3, m(2, 2, 3))
        call intpqdrdy(xs, ys, N3, N2, N3, m(3, 2, 3))
        call intpqdrdy(xs, ys, N4, N2, N3, m(4, 2, 3))

        call intpqdrdy(xs, ys, N1, N3, N3, m(1, 3, 3))
        call intpqdrdy(xs, ys, N2, N3, N3, m(2, 3, 3))
        call intpqdrdy(xs, ys, N3, N3, N3, m(3, 3, 3))
        call intpqdrdy(xs, ys, N4, N3, N3, m(4, 3, 3))

        call intpqdrdy(xs, ys, N1, N4, N3, m(1, 4, 3))
        call intpqdrdy(xs, ys, N2, N4, N3, m(2, 4, 3))
        call intpqdrdy(xs, ys, N3, N4, N3, m(3, 4, 3))
        call intpqdrdy(xs, ys, N4, N4, N3, m(4, 4, 3))


        call intpqdrdy(xs, ys, N1, N1, N4, m(1, 1, 4))
        call intpqdrdy(xs, ys, N2, N1, N4, m(2, 1, 4))
        call intpqdrdy(xs, ys, N3, N1, N4, m(3, 1, 4))
        call intpqdrdy(xs, ys, N4, N1, N4, m(4, 1, 4))

        call intpqdrdy(xs, ys, N1, N2, N4, m(1, 2, 4))
        call intpqdrdy(xs, ys, N2, N2, N4, m(2, 2, 4))
        call intpqdrdy(xs, ys, N3, N2, N4, m(3, 2, 4))
        call intpqdrdy(xs, ys, N4, N2, N4, m(4, 2, 4))

        call intpqdrdy(xs, ys, N1, N3, N4, m(1, 3, 4))
        call intpqdrdy(xs, ys, N2, N3, N4, m(2, 3, 4))
        call intpqdrdy(xs, ys, N3, N3, N4, m(3, 3, 4))
        call intpqdrdy(xs, ys, N4, N3, N4, m(4, 3, 4))

        call intpqdrdy(xs, ys, N1, N4, N4, m(1, 4, 4))
        call intpqdrdy(xs, ys, N2, N4, N4, m(2, 4, 4))
        call intpqdrdy(xs, ys, N3, N4, N4, m(3, 4, 4))
        call intpqdrdy(xs, ys, N4, N4, N4, m(4, 4, 4))

    end subroutine get_rudvdy_matrix

    subroutine matbyvec(a, x, y)
        ! custom subroutine for matrix by vector multiplication to improve performance of Tapenade differentiated
        ! code

        real(8), intent(IN) :: a(1:4, 1:4), x(1:4)
        real(8), intent(INOUT) :: y(1:4)

        integer :: i, j

        do i=1, 4
            do j=1, 4
                y(i)=y(i)+a(i, j)*x(j)
            end do
        end do

    end subroutine matbyvec

    subroutine get_mesh_resmats(ncells, xs, ys, &
        rvj, rdxj, rdyj, rudxj, rudyj)

        integer, intent(IN) :: ncells
        real(8), intent(IN) :: xs(1:ncells, 1:4), ys(1:ncells, 1:4)
        real(8), intent(OUT) :: rvj(1:ncells, 1:4, 1:4), rdxj(1:ncells, 1:4, 1:4), rdyj(1:ncells, 1:4, 1:4), &
        rudxj(1:ncells, 1:4, 1:4, 1:4), rudyj(1:ncells, 1:4, 1:4, 1:4)

        integer :: i

        do i=1, ncells
            call get_rv_matrix(xs(i, :), ys(i, :), rvj(i, :, :))
            call get_rdvdx_matrix(xs(i, :), ys(i, :), rdxj(i, :, :))
            call get_rdvdy_matrix(xs(i, :), ys(i, :), rdyj(i, :, :))
            call get_rudvdx_matrix(xs(i, :), ys(i, :), rudxj(i, :, :, :))
            call get_rudvdy_matrix(xs(i, :), ys(i, :), rudyj(i, :, :, :))
        end do

    end subroutine get_mesh_resmats

    subroutine jac_mult(nnodes, ncells, J, cellmatrix, xd, yd)

        integer, intent(IN) :: nnodes, ncells, cellmatrix(1:ncells, 1:4)
        real(8), intent(IN) :: J(1:ncells, 1:4, 1:4), xd(1:nnodes)
        real(8), intent(OUT) :: yd(1:nnodes)

        integer :: nc, inds(1:4)

        do nc=1, ncells
            inds=cellmatrix(nc, :)

            yd(inds)=yd(inds)+matmul(J(nc, :, :), xd(inds))
        end do

    end subroutine jac_mult

end module residual
