module three_equation

USE ISO_C_BINDING
  IMPLICIT NONE
  REAL*8, PARAMETER :: eps=1e-15, log10eps=&
&   1.0000000000000022, expit_lim=709.7

contains

subroutine uwq(qx, qy, qz, mtosys, u, w, qe)

        real(8), intent(IN) :: qx(1:4), qy(1:4), qz(1:4), mtosys(1:3, 1:3)
        real(8), intent(OUT) :: u(1:4), w(1:4), qe(1:4)

        qe=sqrt(qx**2+qy**2+qz**2)

        u=mtosys(1, 1)*qx+mtosys(1, 2)*qy+mtosys(1, 3)*qz
        w=mtosys(3, 1)*qx+mtosys(3, 2)*qy+mtosys(3, 3)*qz

    end subroutine uwq

    subroutine mache(qe, v_sonic, m)

        real(8), intent(IN) :: qe(1:4), v_sonic
        real(8), intent(OUT) :: M(1:4)

        m=qe/v_sonic

    end subroutine mache

    subroutine rhoe(m, a, rho0, uinf, rho)

        real(8), intent(IN) :: m(1:4), a, rho0, uinf
        real(8), intent(OUT) :: rho(1:4)

        rho=(1.0-m**2*(m*a-uinf)/uinf)*rho0

    end subroutine rhoe

    subroutine reth(qe, rho, th11, mu, re)

        real(8), intent(IN) :: qe(1:4), rho(1:4), th11(1:4), mu
        real(8), intent(OUT) :: re(1:4)

        integer :: i

        re=qe*rho*th11/mu

        do i=1, 4
            if(re(i) .lt. log10eps) then
                re(i)=log10eps
            end if
        end do

    end subroutine reth

    subroutine expit(x, n, sig)

        integer, intent(IN) :: n
        real(8), intent(IN) :: x(1:n)
        real(8), intent(OUT) :: sig(1:n)

        integer :: i

        do i=1, n
            if(x(i) .lt. -expit_lim) then
                sig(i)=0.0
            else if(x(i) .gt. expit_lim) then
                sig(i)=1.0
            else
                sig(i)=1.0/(1.0+dexp(-x(i)))
            end if
        end do

    end subroutine

    subroutine rethcrit(hk, rethc)

        real(8), intent(IN) :: hk(1:4)
        real(8), intent(OUT) :: rethc(1:4)

        rethc=10.0**(3.295/(hk-1.0)+0.44+(1.415/(hk-1.0)-0.489)*dtanh(20.0/(hk-1.0)-12.9))

    end subroutine rethcrit

    subroutine dndreth(rethc, rth, hk, a, dndr)

        real(8), intent(IN) :: rethc(1:4), rth(1:4), hk(1:4), a
        real(8), intent(OUT) :: dndr(1:4)

        real(8) :: sg(1:4)

        call expit(a*(rth-rethc), 4, sg)

        dndr=0.01*sqrt((2.4*hk-3.7+2.5*dtanh(1.5*hk-4.65))**2+0.15)*sg

    end subroutine dndreth

    subroutine p_trans(rth, hk, th11, a, p)

        real(8), intent(IN) :: rth(1:4), hk(1:4), th11(1:4), a
        real(8), intent(OUT) :: p(1:4)

        real(8) :: rthc(1:4), dndr(1:4), m(1:4), l(1:4)

        call rethcrit(hk, rthc)

        call dndreth(rthc, rth, hk, a, dndr)

        l=(6.54*hk-14.07)/hk**2

        m=(0.058*(hk-4.0)**2/(hk-1.0)-0.068)/l

        p=dndr*((m+1.0)/2)*l/th11

    end subroutine p_trans

    subroutine hk_closure(h, me, hk)

        real(8), intent(IN) :: h(1:4), me(1:4)
        real(8), intent(OUT) :: hk(1:4)

        hk=(h-0.290*me**2)/(1.0+0.113*me**2)

    end subroutine hk_closure

    subroutine hstar_laminar(hk, hst)

        real(8), intent(IN) :: hk(1:4)
        real(8), intent(OUT) :: hst(1:4)

        integer :: i

        do i=1, 4
            if(hk(i) .lt. 4.0) then
                hst(i)=0.076*(4.0-hk(i))**2/hk(i)+1.515
            else
                hst(i)=0.04*(hk(i)-4.0)**2/hk(i)+1.515
            end if
        end do

    end subroutine hstar_laminar

    subroutine cf_laminar(hk, rth, cf)

        real(8), intent(IN) :: hk(1:4), rth(1:4)
        real(8), intent(OUT) :: cf(1:4)

        real(8) :: tau(1:4)

        integer :: i

        do i=1, 4
            if(hk(i) .lt. 7.4) then
                tau(i)=0.0396*(7.4-hk(i))**2/(hk(i)-1.0)-0.134
            else
                tau(i)=0.044*(1.0-1.4/(hk(i)-6.0))**2-0.134
            end if
        end do

        cf=tau/rth

    end subroutine cf_laminar

    subroutine hprime_laminar(me, hk, hpr)

        real(8), intent(IN) :: me(1:4), hk(1:4)
        real(8), intent(OUT) :: hpr(1:4)

        hpr=me**2*(0.251+0.064/(hk-0.8))

    end subroutine hprime_laminar

    subroutine cd_laminar(hst, hk, rth, cd)

        real(8), intent(IN) :: hst(1:4), hk(1:4), rth(1:4)
        real(8), intent(OUT) :: cd(1:4)

        real(8) :: D(1:4)

        integer :: i

        do i=1, 4
            if(hk(i) .lt. 4.0) then
                D(i)=0.001025*(4.0-hk(i))**5.5+0.1035
            else
                D(i)=(0.207-0.003*(hk(i)-4.0)**2/(1.0+0.02*(hk(i)-4.0)**2))/2.0
            end if
        end do

        cd=(D*hst)/rth

    end subroutine cd_laminar

    subroutine hstar_turbulent(hk, me, hst)

        real(8), intent(IN) :: hk(1:4), me(1:4)
        real(8), intent(OUT) :: hst(1:4)

        real(8) :: hstme0(1:4)

        hstme0=1.81+3.84*dexp(-2*Hk)-datan((10.0**(7.0-hk)-1.0)/1.23)/8.55-&
        0.146*sqrt(dtanh(2.14*10.0**(4.0-1.46*hk)))

        hst=(hstme0+0.028*me**2)/(1.0+0.014*me**2)

    end subroutine hstar_turbulent

    subroutine hprime_turbulent(me, hk, hpr)

        real(8), intent(IN) :: me(1:4), hk(1:4)
        real(8), intent(OUT) :: hpr(1:4)

        hpr=me**2*(0.251+0.064/(hk-0.8))

    end subroutine hprime_turbulent

    subroutine fc(me, gamma, f)

        real(8), intent(IN) :: me(1:4), gamma
        real(8), intent(OUT) :: f(1:4)

        f=sqrt(1.0+(gamma-1.0)*me**2/2)

    end subroutine fc

    subroutine cf_turbulent(hk, rth, f, cf)

        real(8), intent(IN) :: hk(1:4), rth(1:4), f(1:4)
        real(8), intent(OUT) :: cf(1:4)

        real(8) :: cf_bar(1:4)

        cf_bar=0.3*(log10(rth))**(-0.31*hk-1.74)*dexp(-1.33*hk)+&
        0.00011*(dtanh(4.0-8.0*hk/7.0)-1.0)

        cf=cf_bar/f

    end subroutine cf_turbulent

    subroutine cd_turbulent(hk, f, me, rth, cd)

        real(8), intent(IN) :: hk(1:4), f(1:4), me(1:4), rth(1:4)
        real(8), intent(OUT) :: cd(1:4)

        real(8) :: a(1:4), b(1:4), c(1:4)
        integer :: i

        do i=1, 4
            if(hk(i) .lt. 3.5) then
                a(i)=0.160*(hk(i)-3.5)-0.550
            else
                a(i)=0.438-0.280*hk(i)
            end if
        end do

        b=0.009-0.011*dexp(-0.15*hk**2.1)+3e-5*dexp(0.117*hk**2)

        c=f*(1.0+0.05*me**1.4)

        cd=2*(b+a*rth**(-0.574))/c

    end subroutine cd_turbulent

    subroutine sigma_n(nts, a, ncrit, sn)

        real(8), intent(IN) :: nts(1:4), a, ncrit
        real(8), intent(OUT) :: sn(1:4)

        call expit(a*(nts-ncrit), 4, sn)

    end subroutine sigma_n

    subroutine a_crossflow(cf, beta, me, a)

        real(8), intent(IN) :: cf(1:4), beta(1:4), me(1:4)
        real(8), intent(OUT) :: a(1:4)

        real(8) :: g(1:4)

        g=sqrt(cf*cos(beta)*(1.0+0.18*me**2))

        a=dtan(beta)*(g/(g-1.0)+1.0)

    end subroutine a_crossflow

    subroutine deltastar_innode(th11, h, a, deltastar_1, deltastar_2)

        real(8), intent(IN) :: th11(1:4), h(1:4), a(1:4)
        real(8), intent(OUT) :: deltastar_1(1:4), deltastar_2(1:4)

        deltastar_1=h*th11

        deltastar_2=-A*deltastar_1

    end subroutine deltastar_innode

    subroutine theta_innode(th11, a, deltastar_2, th12, th21, th22)

        real(8), intent(IN) :: th11(1:4), a(1:4), deltastar_2(1:4)
        real(8), intent(OUT) :: th12(1:4), th21(1:4), th22(1:4)

        th21=-a*th11
        th12=th21-deltastar_2
        th22=-a*th12

    end subroutine theta_innode

    subroutine thetastar_innode(hst, a, deltastar_1, th11, th22, thst1, thst2)

        real(8), intent(IN) :: hst(1:4), a(1:4), deltastar_1(1:4), th11(1:4), th22(1:4)
        real(8), intent(OUT) :: thst1(1:4), thst2(1:4)

        thst1=hst*th11
        thst2=a*(deltastar_1+th11+th22-thst1)

    end subroutine thetastar_innode

    subroutine deltaprime_innode(hpr, a, th11, deltaprime_1, deltaprime_2)

        real(8), intent(IN) :: hpr(1:4), a(1:4), th11(1:4)
        real(8), intent(OUT) :: deltaprime_1(1:4), deltaprime_2(1:4)

        deltaprime_1=hpr*th11
        deltaprime_2=-A*deltaprime_1

    end subroutine deltaprime_innode

    subroutine cd_innode(cd, a, cd_2)

        real(8), intent(IN) :: cd(1:4), a(1:4)
        real(8), intent(OUT) :: cd_2(1:4)

        real(8) :: a_aux(1:4), base, lbase, eden, enum, expon
        integer :: i

        a_aux=dabs(a)

        do i=1, 4
            base=14667.0*cd(i)+3.0

            if(a_aux(i) .gt. 1e-15) then
                lbase=log(base)

                eden=1020.0*cd(i)+4.0

                enum=a(i)+10.0

                expon=enum/eden

                cd_2=base**expon
            end if
        end do

    end subroutine cd_innode

    subroutine j_innode(th11, th12, th21, th22, u, w, rho, jxx, jxz, jzx, jzz)

        real(8), intent(IN) :: th11(1:4), th12(1:4), th21(1:4), th22(1:4), u(1:4), w(1:4), rho(1:4)
        real(8), intent(OUT) :: jxx(1:4), jxz(1:4), jzx(1:4), jzz(1:4)

        real(8) :: u2(1:4), uw(1:4), w2(1:4)

        u2=u**2
        uw=u*w
        w2=w**2

        jxx=(u2*th11-uw*th12-uw*th21+w2*th22)*rho
        jxz=(uw*th11+u2*th12-w2*th21-uw*th22)*rho
        jzx=(uw*th11-w2*th12+u2*th21-uw*th22)*rho
        jzz=(w2*th11+uw*th12+uw*th21+u2*th22)*rho

    end subroutine j_innode

    subroutine m_innode(deltastar_1, deltastar_2, u, w, rho, mx, mz)

        real(8), intent(IN) :: deltastar_1(1:4), deltastar_2(1:4), u(1:4), w(1:4), rho(1:4)
        real(8), intent(OUT) :: mx(1:4), mz(1:4)

        mx=(deltastar_1*u+deltastar_2*w)*rho
        mz=(deltastar_2*u-deltastar_1*w)*rho

    end subroutine m_innode

    subroutine e_innode(thst1, thst2, u, w, q, rho, ex, ez)

        real(8), intent(IN) :: thst1(1:4), thst2(1:4), u(1:4), w(1:4), q(1:4), rho(1:4)
        real(8), intent(OUT) :: ex(1:4), ez(1:4)

        ex=rho*q**2*(thst1*u+thst2*w)
        ez=rho*q**2*(thst2*u-thst1*w)

    end subroutine e_innode

    subroutine rhoq_innode(deltaprime_1, deltaprime_2, u, w, rho, rhoqx, rhoqz)

        real(8), intent(IN) :: deltaprime_1(1:4), deltaprime_2(1:4), u(1:4), w(1:4), rho(1:4)
        real(8), intent(OUT) :: rhoqx(1:4), rhoqz(1:4)

        rhoqx=rho*(deltaprime_1*u+deltaprime_2*w)
        rhoqz=rho*(deltaprime_2*u-deltaprime_1*w)

    end subroutine rhoq_innode

    subroutine tau_innode(cf_1, cf_2, u, w, q, rho, taux, tauz)

        real(8), intent(IN) :: cf_1(1:4), cf_2(1:4), u(1:4), w(1:4), q(1:4), rho(1:4)
        real(8), intent(OUT) :: taux(1:4), tauz(1:4)

        taux=rho*(cf_1*q*u+cf_2*q*w)/2
        tauz=rho*(cf_2*q*u-cf_2*q*w)/2

    end subroutine tau_innode

    subroutine matbyvec(a, x, y)
        ! custom subroutine for matrix by vector multiplication to improve performance of Tapenade differentiated
        ! code

        real(8), intent(IN) :: a(1:4, 1:4), x(1:4)
        real(8), intent(INOUT) :: y(1:4)

        y=y+matmul(a, x)

    end subroutine matbyvec

    subroutine mat3byvec(a, x, y, z)
        ! custom subroutine for rank 3 matrix by vectors multiplication to improve performance of Tapenade differentiated
        ! code

        real(8), intent(IN) :: a(1:4, 1:4, 1:4), x(1:4), y(1:4)
        real(8), intent(INOUT) :: z(1:4)

        real(8) :: interm(1:4)

        integer :: i, j

        do i=1, 4
            interm=matmul(a(i, :, :), y)

            z(i)=z(i)+dot_product(interm, x)
        end do

    end subroutine mat3byvec

    subroutine cell_getresiduals(n, th11, h, beta, nts, qx, qy, qz, &
        rho0, v_sonic, a_transition, a_rethcrit, &
        mtosys, uinf, mu, ncrit, gamma, rvj, rdxj, rdyj, rudxj, rudyj, &
        rmass, rmomx, rmomz, ren, rts)

        real(8), intent(IN) :: n(1:4), th11(1:4), h(1:4), beta(1:4), nts(1:4), qx(1:4), qy(1:4), qz(1:4)
        real(8), intent(IN) :: rho0, v_sonic, a_transition, a_rethcrit, mtosys(1:3, 1:3), uinf, mu, &
        ncrit, gamma, rvj(1:4, 1:4), rdxj(1:4, 1:4), rdyj(1:4, 1:4), rudxj(1:4, 1:4, 1:4), rudyj(1:4, 1:4, 1:4)
        real(8), intent(OUT) :: rmass(1:4), rmomx(1:4), rmomz(1:4), ren(1:4), rts(1:4)
        ! residual matrixes: rows for residual number

        real(8) :: qe(1:4), u(1:4), w(1:4), m(1:4), rho(1:4), rth(1:4), hk(1:4), &
        p(1:4), sgn(1:4), cl(1:4), ct(1:4), cf(1:4), f(1:4), a_cr(1:4), &
        cf_cr(1:4), hst(1:4), hpr(1:4), cd(1:4), cd_cr(1:4), taux(1:4), tauz(1:4), &
        D(1:4), dst1(1:4), dst2(1:4), th12(1:4), th21(1:4), th22(1:4), dpr1(1:4), dpr2(1:4), &
        thst1(1:4), thst2(1:4), jxx(1:4), jxz(1:4), jzx(1:4), jzz(1:4), &
        mx(1:4), mz(1:4), ex(1:4), ez(1:4), rhoqx(1:4), rhoqz(1:4)

        call uwq(qx, qy, qz, mtosys, u, w, qe)
        call mache(qe, v_sonic, m)
        call rhoe(m, v_sonic, rho0, uinf, rho)
        call reth(qe, rho, th11, mu, rth)

        call hk_closure(h, m, hk)

        call p_trans(rth, hk, th11, a_rethcrit, p)
        call sigma_n(nts, ncrit, a_transition, sgn)

        call fc(m, gamma, f)

        ! closure relationships
        call cf_laminar(hk, rth, cl)
        call cf_turbulent(hk, rth, f, ct)

        cf=sgn*ct+(1.0-sgn)*cl

        call a_crossflow(cf, beta, m, a_cr)

        cf_cr=-cf*dtan(beta)

        call hstar_laminar(hk, cl)
        call hstar_turbulent(hk, m, ct)

        hst=sgn*ct+(1.0-sgn)*cl

        call hprime_laminar(m, hk, cl)
        call hprime_turbulent(m, hk, ct)

        hpr=sgn*ct+(1.0-sgn)*cl

        call cd_laminar(hst, hk, rth, cl)
        call cd_turbulent(hk, f, m, rth, ct)

        cd=sgn*ct+(1.0-sgn)*cl

        call cd_innode(cd, a_cr, cd_cr)

        D=rho*qe**3*(cd+cd_cr)

        call tau_innode(cf, cf_cr, u, w, qe, rho, taux, tauz)

        call deltastar_innode(th11, h, a_cr, dst1, dst2)
        call theta_innode(th11, a_cr, dst2, th12, th21, th22)
        call deltaprime_innode(hpr, a_cr, th11, dpr1, dpr2)
        call thetastar_innode(hst, a_cr, dst1, th11, th22, thst1, thst2)

        call j_innode(th11, th12, th21, th22, u, w, rho, jxx, jxz, jzx, jzz)
        call m_innode(dst1, dst2, u, w, rho, mx, mz)
        call e_innode(thst1, thst2, u, w, qe, rho, ex, ez)
        call rhoq_innode(dpr1, dpr2, u, w, rho, rhoqx, rhoqz)

        ! initialize residuals
        rmass=0.0
        rmomx=0.0
        rmomz=0.0
        rts=0.0
        ren=0.0

        call matbyvec(rdxj, jxx, rmomx)
        call matbyvec(rdyj, jxz, rmomx)
        call matbyvec(rvj, -taux, rmomx)
        call mat3byvec(rudxj, mx, u, rmomx)
        call mat3byvec(rudyj, mz, u, rmomx)

        call matbyvec(rdxj, jzx, rmomz)
        call matbyvec(rdyj, jzz, rmomz)
        call matbyvec(rvj, -tauz, rmomz)
        call mat3byvec(rudxj, mx, w, rmomz)
        call mat3byvec(rudyj, mz, w, rmomz)

        call matbyvec(rdxj, ex, ren)
        call matbyvec(rdyj, ez, ren)
        call matbyvec(rvj, -2*D, ren)
        call mat3byvec(rudxj, rhoqx, qe**2, ren)
        call mat3byvec(rudyj, rhoqz, qe**2, ren)

        call mat3byvec(rudxj, u, nts, rts)
        call mat3byvec(rudyj, w, nts, rts)
        call matbyvec(rvj, -qe*p, rts)

        call matbyvec(rdxj, mx, rmass)
        call matbyvec(rdyj, mz, rmass)
        call matbyvec(rvj, -rho*n, rmass)

    end subroutine cell_getresiduals

    subroutine mesh_getresiduals(nnodes, ncells, cellmat, n, th11, h, beta, nts, qx, qy, qz, &
        rho0, v_sonic, a_transition, a_rethcrit, &
        mtosys, uinf, mu, ncrit, gamma, rvj, rdxj, rdyj, rudxj, rudyj, &
        rmass, rmomx, rmomz, ren, rts)

        integer, intent(IN) :: nnodes, ncells
        integer, intent(IN) :: cellmat(1:ncells, 1:4)

        real(8), intent(IN) :: n(1:nnodes), th11(1:nnodes), h(1:nnodes), beta(1:nnodes), &
        nts(1:nnodes), qx(1:nnodes), qy(1:nnodes), qz(1:nnodes), rho0, v_sonic, a_transition, a_rethcrit, &
        mtosys(1:ncells, 1:3, 1:3), uinf, mu, ncrit, gamma, rvj(1:ncells, 1:4, 1:4), rdxj(1:ncells, 1:4, 1:4), &
        rdyj(1:ncells, 1:4, 1:4), rudxj(1:ncells, 1:4, 1:4, 1:4), rudyj(1:ncells, 1:4, 1:4, 1:4)

        real(8), intent(OUT) :: rmass(1:nnodes), rmomx(1:nnodes), rmomz(1:nnodes), ren(1:nnodes), rts(1:nnodes)

        integer :: i, inds(1:4)
        real(8) :: rmass_l(1:4), rmomx_l(1:4), rmomz_l(1:4), ren_l(1:4), rts_l(1:4)

        rmass=0.0
        rmomx=0.0
        rmomz=0.0
        ren=0.0
        rts=0.0

        do i=1, ncells
            inds=cellmat(i, :)

            call cell_getresiduals(n(inds), th11(inds), h(inds), beta(inds), nts(inds), &
                qx(inds), qy(inds), qz(inds), &
                rho0, v_sonic, a_transition, a_rethcrit, &
                mtosys(i, :, :), uinf, mu, ncrit, gamma, rvj(i, :, :), rdxj(i, :, :), &
                rdyj(i, :, :), rudxj(i, :, :, :), rudyj(i, :, :, :), &
                rmass_l, rmomx_l, rmomz_l, ren_l, rts_l)

            rmass(inds)=rmass(inds)+rmass_l
            rmomx(inds)=rmomx(inds)+rmomx_l
            rmomz(inds)=rmomz(inds)+rmomz_l
            ren(inds)=ren(inds)+ren_l
            rts(inds)=rts(inds)+rts_l
        end do

    end subroutine mesh_getresiduals

end module three_equation
