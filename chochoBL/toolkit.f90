! FILE CONTAINING DEFINITIONS FOR MATHEMATICAL OPERATIONS IN FORTRAN BACKEND
! NOW CONTAINING
! 1- for internal use: quick inversion of coordinate system matrixes
! 2- quick orthonormalization of linear coordinate systems
! 3- set local streamwise coordinate system based on local velocities and normal vectors
! 4- differentiate property with respect to lambda_x and lambda_y auxiliary coordinates
! 5- create matrix for surface gradient calculation
! 6- compute surface gradients based on surface gradient calculation matrix (generated in 5)
! 7- compute derivatives in normal direction y
! 8- quickly compute v velocity component based on pressure gradient, previously computed z derivatives
! and y second order derivative of u
! 9- put BL equations into exercise so as to compute velocity gradients
! 10- propagate a velocity gradient from a row to another
! 11- quickly integrating a property along normal direction of all grid cells
! 12- quickly perform attachment check based on Goldstein's singularity

subroutine invert3(a, det_l) !coordinate system matrix inverter
    double precision, intent(inout) :: a (1:3, 1:3)
    real(8), intent(out) :: det_l
    real(8) :: b(1:3, 1:3)

    det_l = a(1,1)*(a(2,2)*a(3,3)-a(2,3)*a(3,2)) &
        -a(1,2)*(a(2,1)*a(3,3)-a(2,3)*a(3,1)) &
        +a(1,3)*(a(2,1)*a(3,2)-a(2,2)*a(3,1))
    
    b=a

    a(1,1) =  b(2,2)*b(3,3) - b(2,3)*b(3,2)
    a(2,1) =  b(2,3)*b(3,1) - b(2,1)*b(3,3)
    a(3,1) =  b(2,1)*b(3,2) - b(2,2)*b(3,1)

    a(1,2) =  b(1,3)*b(3,2) - b(1,2)*b(3,3)
    a(2,2) =  b(1,1)*b(3,3) - b(1,3)*b(3,1)
    a(3,2) =  b(1,2)*b(3,1) - b(1,1)*b(3,2)

    a(1,3) =  b(1,2)*b(2,3) - b(1,3)*b(2,2)
    a(2,3) =  b(1,3)*b(2,1) - b(1,1)*b(2,3)
    a(3,3) =  b(1,1)*b(2,2) - b(1,2)*b(2,1)

    a=a/det_l
end

subroutine orthon(u, v, Mtosys, Mtouni)
    real(8), intent(IN) :: u(1:3), v(1:3)
    real(8), intent(OUT) :: Mtosys(1:3, 1:3), Mtouni(1:3, 1:3)

    Mtosys(1, 1:3)=u/norm2(u)
    Mtosys(2, 1:3)=v-dot_product(Mtosys(1, 1:3), v)*Mtosys(1, 1:3)
    Mtosys(2, 1:3)=Mtosys(2, 1:3)/norm2(Mtosys(2, 1:3))
    Mtosys(3, 1:3)=(/Mtosys(1, 2)*Mtosys(2, 3)-Mtosys(1, 3)*Mtosys(2, 2), &
    Mtosys(1, 3)*Mtosys(2, 1)-Mtosys(1, 1)*Mtosys(2, 3), &
    Mtosys(1, 1)*Mtosys(2, 2)-Mtosys(1, 2)*Mtosys(2, 1)/)
    Mtouni=transpose(Mtosys)
end subroutine orthon

subroutine mtosys_gen(idisc, jdisc, ues, ves, wes, nvects, Mtosys, Mtouni)
    integer, intent(IN) :: idisc, jdisc
    real(8), intent(IN) :: ues(1:idisc, 1:jdisc), ves(1:idisc, 1:jdisc), wes(1:idisc, 1:jdisc)
    real(8), intent(INOUT) :: nvects(1:idisc, 1:jdisc, 1:3)
    real(8), intent(OUT) :: Mtosys(1:idisc, 1:jdisc, 1:3, 1:3), Mtouni(1:idisc, 1:jdisc, 1:3, 1:3)

    real(8) :: qe(1:3), nv(1:3), latvec(1:3)
    integer :: i, j

    do i=1, idisc
        do j=1, jdisc
            qe=(/ues(i, j), ves(i, j), wes(i, j)/)
            nv=nvects(i, j, 1:3)
            nv=nv/norm2(nv)
            nvects(i, j, 1:3)=nv
            qe=qe-dot_product(qe, nv)*nv
            qe=qe/norm2(qe)
            latvec=(/nv(3)*qe(2)-nv(2)*qe(3), nv(1)*qe(3)-nv(3)*qe(1), nv(2)*qe(1)-nv(1)*qe(2)/)
            Mtosys(i, j, 1, 1:3)=qe
            Mtosys(i, j, 3, 1:3)=latvec
            Mtosys(i, j, 2, 1:3)=nv
            Mtouni(i, j, 1:3, 1:3)=transpose(Mtosys(i, j, 1:3, 1:3))
        end do
    end do
end subroutine Mtosys_gen

subroutine lambda_grad(idisc, jdisc, prop, dlx, dly, lgrad)
    integer, intent(IN) :: idisc, jdisc
    real(8), intent(IN) :: prop(1:idisc, 1:jdisc), dlx, dly
    real(8), intent(OUT) :: lgrad(1:idisc, 1:jdisc, 1:2)

    lgrad(1, :, 1)=(prop(2, :)-prop(1, :))/dlx
    lgrad(2:idisc-1, :, 1)=(prop(3:idisc, :)-prop(1:idisc-2, :))/(2*dlx)
    lgrad(idisc, :, 1)=(prop(idisc, :)-prop(idisc-1, :))/dlx

    lgrad(:, 1, 2)=(prop(:, 2)-prop(:, 1))/dly
    lgrad(:, 2:jdisc-1, 2)=(prop(:, 3:jdisc)-prop(:, 1:jdisc-2))/(2*dly)
    lgrad(:, jdisc, 2)=(prop(:, jdisc)-prop(:, jdisc-1))/dly
end subroutine lambda_grad

subroutine surfgradmat(idisc, jdisc, Mtosys, dxdlx, dydlx, dzdlx, dxdly, dydly, dzdly, gradmat)
    integer, intent(IN) :: idisc, jdisc
    real(8), intent(IN) :: Mtosys(1:idisc, 1:jdisc, 1:3, 1:3), &
    dxdlx(1:idisc, 1:jdisc), dydlx(1:idisc, 1:jdisc), dzdlx(1:idisc, 1:jdisc), dxdly(1:idisc, 1:jdisc), &
    dydly(1:idisc, 1:jdisc), dzdly(1:idisc, 1:jdisc)
    real(8), intent(OUT) :: gradmat(1:idisc, 1:jdisc, 1:3, 1:2)

    integer :: i, j
    real(8) :: M(1:3, 1:3), det_l

    do i=1, idisc
        do j=1, jdisc
            M(1, :)=(/dxdlx(i, j), dydlx(i, j), dzdlx(i, j)/)
            M(2, :)=(/dxdly(i, j), dydly(i, j), dzdly(i, j)/)
            M(3, :)=Mtosys(i, j, 2, :)
            call invert3(M, det_l)
            M=matmul(Mtosys(i, j, :, :), M)
            gradmat(i, j, :, :)=M(:, 1:2)
        end do
    end do
end subroutine surfgradmat

subroutine calcgrad(idisc, jdisc, Mtosys, propderiv, gradmat, tosys, grad) !if tosys is .TRUE., gradients returned in local coordinate system
    integer, intent(IN) :: idisc, jdisc
    real(8), intent(IN) :: Mtosys(1:idisc, 1:jdisc, 1:3, 1:3), propderiv(1:idisc, 1:jdisc, 1:2), gradmat(1:idisc, 1:jdisc, 1:3, 1:2)
    logical, intent(IN) :: tosys
    real(8), intent(OUT) :: grad(1:idisc, 1:jdisc, 1:3)

    integer :: i, j

    do i=1, idisc
        do j=1, jdisc
            grad(i, j, :)=matmul(gradmat(i, j, :, :), propderiv(i, j, :))
            if(tosys) then
                grad(i, j, :)=matmul(Mtosys(i, j, :, :), grad(i, j, :))
            end if
        end do
    end do
end subroutine calcgrad

subroutine calcnormgrad(jdisc, ndisc, prop, thicks, dlt, dydlt, grad)
    integer, intent(IN) :: jdisc, ndisc
    real(8), intent(IN) :: prop(1:jdisc, 1:ndisc), thicks(1:jdisc, 1:ndisc), dlt, dydlt(1:ndisc)
    real(8), intent(OUT) :: grad(1:jdisc, 1:ndisc)

    integer :: n

    !computing local dlambda_t derivative
    grad(:, 2:ndisc-1)=(prop(:, 3:ndisc)-prop(:, 1:ndisc-2))/(2*dlt)
    grad(:, 1)=(prop(:, 2)-prop(:, 1))/dlt
    grad(:, ndisc)=(prop(:, ndisc)-prop(:, ndisc-1))/dlt

    !computing local dy derivative
    do n=1, ndisc
        grad(:, n)=grad(:, n)/(dydlt(n)*thicks(:, ndisc))
    end do
end subroutine calcnormgrad

subroutine calcv(jdisc, ndisc, dudz, dwdz, us, ws, dpdx, mu, rho, d2udy2, thicks, vs)
    integer, intent(IN) :: jdisc, ndisc
    real(8), intent(IN) :: dudz(1:jdisc, 1:ndisc), dwdz(1:jdisc, 1:ndisc), us(1:jdisc, 1:ndisc), ws(1:jdisc, 1:ndisc), &
    dpdx(1:jdisc), mu, rho, d2udy2(1:jdisc, 1:ndisc), thicks(1:jdisc, 1:ndisc) !nu denoting kinematic viscosity
    real(8), intent(OUT) :: vs(1:jdisc, 1:ndisc-2)

    real(8) :: intvec(1:jdisc, 1:ndisc-2), previous(1:jdisc), integrated(1:jdisc, 1:ndisc-2)
    integer :: n

    intvec=us(:, 2:ndisc-1)*dwdz(:, 2:ndisc-1)-ws(:, 2:ndisc-1)*dudz(:, 2:ndisc-1)+(mu/rho)*d2udy2(:, 2:ndisc-1)
    do n=1, ndisc-2
        intvec(:, n)=intvec(:, n)-dpdx
    end do
    intvec=-intvec/(us(:, 2:ndisc-1)**2)

    !integrating on y direction
    previous=0.0
    do n=1, ndisc-2
        integrated(:, n)=previous+intvec(:, n)*(thicks(:, n+1)-thicks(:, n))
        previous=integrated(:, n)
    end do
    vs=integrated*us(:, 2:ndisc-1)
end subroutine calcv

subroutine blexercise(jdisc, ndisc, mu, rho, us, vs, ws, dudz, dwdz, dudy, dwdy, pressgrad, d2udy2, d2wdy2, dudx, dwdx)
    integer, intent(IN) :: jdisc, ndisc
    real(8), intent(IN) :: mu, rho, us(1:jdisc, 1:ndisc), vs(1:jdisc, 1:ndisc), ws(1:jdisc, 1:ndisc), &
    dudz(1:jdisc, 1:ndisc), dwdz(1:jdisc, 1:ndisc), &
    dudy(1:jdisc, 1:ndisc), dwdy(1:jdisc, 1:ndisc), &
    pressgrad(1:jdisc, 1:3), &
    d2udy2(1:jdisc, 1:ndisc), d2wdy2(1:jdisc, 1:ndisc)
    real(8), intent(OUT) :: dudx(1:jdisc, 1:ndisc-2), dwdx(1:jdisc, 1:ndisc-2)

    integer :: n

    dudx=(mu/rho)*d2udy2(:, 2:ndisc-1)-vs(:, 2:ndisc-1)*dudy(:, 2:ndisc-1)-ws(:, 2:ndisc-1)*dudz(:, 2:ndisc-1)
    dwdx=(mu/rho)*d2wdy2(:, 2:ndisc-1)-vs(:, 2:ndisc-1)*dwdy(:, 2:ndisc-1)-ws(:, 2:ndisc-1)*dwdz(:, 2:ndisc-1)
    do n=1, ndisc-2
        dudx(:, n)=dudx(:, n)-pressgrad(:, 1)/rho
        dwdx(:, n)=dwdx(:, n)-pressgrad(:, 3)/rho
    end do
    dudx=dudx/us(:, 2:ndisc-1)
    dwdx=dwdx/us(:, 2:ndisc-1)
end subroutine blexercise

subroutine blpropagate(jdisc, ndisc, p0_1, p0_2, thicks_1, thicks_2, us, ws, Mtosys_1, Mtosys_2, &
Mtouni_1, Mtouni_2, dudx, dwdx, dudy, dwdy, dudz, dwdz, newrow_us, newrow_ws)
    integer, intent(IN) :: jdisc, ndisc
    real(8), intent(IN) :: p0_1(1:jdisc, 1:3), p0_2(1:jdisc, 1:3), thicks_1(1:jdisc, 1:ndisc), thicks_2(1:jdisc, 1:ndisc), &
    us(1:jdisc, 1:ndisc), ws(1:jdisc, 1:ndisc), &
    Mtosys_1(1:jdisc, 1:3, 1:3), Mtosys_2(1:jdisc, 1:3, 1:3), Mtouni_1(1:jdisc, 1:3, 1:3), Mtouni_2(1:jdisc, 1:3, 1:3), &
    dudx(1:jdisc, 1:ndisc-2), dwdx(1:jdisc, 1:ndisc-2), dudy(1:jdisc, 1:ndisc-2), dwdy(1:jdisc, 1:ndisc-2), &
    dudz(1:jdisc, 1:ndisc-2), dwdz(1:jdisc, 1:ndisc-2)
    real(8), intent(OUT) :: newrow_us(1:jdisc, 1:ndisc-2), newrow_ws(1:jdisc, 1:ndisc-2)

    real(8) :: offset(1:jdisc, 1:ndisc, 1:3), newv(1:jdisc, 1:ndisc-2, 1:2), newv_aux(1:jdisc, 1:ndisc-2, 1:3), n2(1:3), dp(1:3)
    integer :: j, n

    !calculating offsets between points in rows, in R1's local coordinate system
    do j=1, jdisc
        n2=matmul(Mtosys_1(j, :, :), Mtosys_2(j, 2, :))
        dp=matmul(Mtosys_1(j, :, :), p0_2(j, :)-p0_1(j, :))
        do n=1, ndisc
            offset(j, n, :)=dp+n2*thicks_2(j, n)
        end do
        offset(j, :, 2)=offset(j, :, 2)-thicks_1(j, :)
    end do
    newv(:, :, 1)=offset(:, 2:ndisc-1, 1)*dudx+offset(:, 2:ndisc-1, 2)*dudy+offset(:, 2:ndisc-1, 3)*dudz+us(:, 2:ndisc-1)
    newv(:, :, 2)=offset(:, 2:ndisc-1, 1)*dwdx+offset(:, 2:ndisc-1, 2)*dwdy+offset(:, 2:ndisc-1, 3)*dwdz+ws(:, 2:ndisc-1)
    !adding up to compute new velocities
    !transfer calculated velocity variations to new row's coordinate systems
    do j=1, jdisc
        newv_aux(j, :, 1)=newv(j, :, 1)*Mtouni_1(j, 1, 1)+newv(j, :, 2)*Mtouni_1(j, 1, 3)
        newv_aux(j, :, 2)=newv(j, :, 1)*Mtouni_1(j, 2, 1)+newv(j, :, 2)*Mtouni_1(j, 2, 3)
        newv_aux(j, :, 3)=newv(j, :, 1)*Mtouni_1(j, 3, 1)+newv(j, :, 2)*Mtouni_1(j, 3, 3)
        newrow_us(j, :)=newv_aux(j, :, 1)*Mtosys_2(j, 1, 1)+newv_aux(j, :, 2)*Mtosys_2(j, 1, 2)+newv_aux(j, :, 3)*Mtosys_2(j, 1, 3)
        newrow_ws(j, :)=newv_aux(j, :, 1)*Mtosys_2(j, 3, 1)+newv_aux(j, :, 2)*Mtosys_2(j, 3, 2)+newv_aux(j, :, 3)*Mtosys_2(j, 3, 3)
    end do
end subroutine blpropagate

subroutine intthick(idisc, jdisc, ndisc, prop, thicks, dydlt, dlt, propint)
    integer, intent(IN) :: idisc, jdisc, ndisc
    real(8), intent(IN) :: prop(1:idisc, 1:jdisc, 1:ndisc), thicks(1:idisc, 1:jdisc, 1:ndisc), dydlt(1:ndisc), dlt
    real(8), intent(OUT) :: propint(1:idisc, 1:jdisc)

    real(8) :: vecaux(1:ndisc)
    integer :: i, j

    do i=1, idisc
        do j=1, jdisc
            vecaux=prop(i, j, :)*dydlt
            propint(i, j)=thicks(i, j, ndisc)*dlt*sum(vecaux(1:ndisc-1)+vecaux(2:ndisc))/2
        end do
    end do
end subroutine intthick

subroutine checkattach(jdisc, ndisc, us, isattached)
    integer, intent(IN) :: jdisc, ndisc
    real(8), intent(IN) :: us(1:jdisc, 1:ndisc)
    logical, intent(OUT) :: isattached(1:jdisc)

    integer :: j
    
    do j=1, jdisc
        isattached(j)=(.NOT. any(us(j, 2:ndisc)<0.0))
    end do
end subroutine checkattach