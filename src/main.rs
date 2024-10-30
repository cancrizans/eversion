use hexasphere::shapes::IcoSphere;

use three_d::*;

const Q : f32 = 0.66666;

fn eversion_cyl(t : f32, phi : f32, theta : f32, n : usize, omega: f32, lambda : f32, eta : f32, kappa:f32) -> Vector3<f32>{
    let sinth = theta.sin();
    let cosnth = theta.cos().powi(n as i32);



    let sin = phi.sin();
    let cos = phi.cos();
    let nf = n as f32;
    let nmof = nf - 1.0;

    
    let p = 1.0 - (Q*t).abs();

    if t.abs() * Q < 0.999{
        let h = omega * sinth / cosnth;
        Vector3::new(
            t * cos + p*(nmof*phi).sin() - h * sin,
            t * sin + p*(nmof * phi).cos() + h *cos,
            h * (nf * phi).sin() - (t/nf) * ((nf*phi).cos()) - Q*t*h
        )
    } else {
        let lpol = 1.0 -lambda + lambda * cosnth;
        let lomega = lambda * omega;

        let nphi = (n as f32) * phi;
        

        Vector3::new(
            (t*lpol * cos - lomega * sinth * sin) / cosnth,
            (t*lpol * sin + lomega * sinth * cos) / cosnth,
            lambda * (omega * sinth * (nphi.sin() - Q*t) / cosnth - (t/nf) * nphi.cos()) 
                - (1.0 - lambda) * eta.powf(1.0+kappa) * t * t.abs().powf(2.0*kappa) * sinth / (cosnth*cosnth)
        )
    }
    // Vector3::new(
    //     t * cos - h * sin,
    //     t * sin + h *cos,
    //     h * (nf * phi).sin() - (t/nf) * ((nf*phi).cos()) - h
    // )
}

fn damping(r : Vector3<f32>, eta : f32, kappa:f32, xi : f32) -> Vector3<f32>{
    
    let term = xi + eta * (r.x*r.x + r.y*r.y);
    let pow = term.powf(-kappa);
    Vector3::new(r.x * pow, r.y*pow, r.z / term)
}

fn stereo(rprime : Vector3<f32>, alpha: f32, beta:f32) -> Vector3<f32>{
    let gamma = 2.0 * (alpha*beta).sqrt();
    let betarho2 = beta * (rprime.x*rprime.x + rprime.y*rprime.y);
    let expgz = (rprime.z * gamma).exp();

    Vector3::new(
        rprime.x * expgz / (alpha + betarho2),
        rprime.y * expgz / (alpha + betarho2),
        ((alpha - betarho2) / (alpha + betarho2) * expgz - (alpha-beta)/(alpha+beta) )/ gamma
    )
}


fn make_surface(lambda : f32, n : usize, omega: f32, t : f32, eta : f32, xi : f32, alpha: f32, beta:f32) -> CpuMesh{
    
    let ico = IcoSphere::new(23,|_|());
    let points = ico.raw_points();
    let indices = ico.get_all_indices();

    let kappa = (n as f32 - 1.0)/(2.0 * n as f32);

    // const N_PHI : u32 = 64;
    // const N_H : u32 = 32;
    // const VERTCOUNT : u32 = (N_PHI)*(N_H);
    // let mut verts = Vec::with_capacity(VERTCOUNT as usize);
    // let mut indices : Vec<u32> = Vec::with_capacity((VERTCOUNT * 2) as usize);

    // (0..N_PHI).for_each(|i|{
    //     let phi = (i as f32) / (N_PHI as f32) * f32::consts::PI * 2.0;
    //     let next_i_off = if i == N_PHI - 1 {1i32-N_PHI as i32} else {1};
    //     (0..N_H).for_each(|j|{
    //         let theta = ((j as f32) / (N_H as f32) * 2.0 - 1.0) * f32::consts::PI / 2.0;
    //         let next_j_off = if j == N_H - 1 {1i32-N_H as i32} else {1};
            // let h = omega * theta.sin() / theta.cos().powi(n as i32);

    let (verts,_uvs) : (Vec<Vector3<f32>>,Vec<Vector2<f32>>) 
    = points.into_iter().map(|p|{
        let phi = p.y.atan2(p.x);
        let theta = p.z.atan2((p.x*p.x+p.y*p.y).sqrt());

        let r = eversion_cyl(
            t,phi,theta,n,
            omega,lambda,eta,kappa
        );
        let rprime = damping(r, eta, kappa, xi);
        let rsecond = stereo(rprime, alpha, beta);

        let uv = Vector2::new(phi, theta);

        (rsecond,uv)
    }).unzip();

    let mut mesh = CpuMesh{
        positions : Positions::F32(verts),
        indices : Indices::U32(indices),
        // uvs : Some(uvs),
        ..Default::default()
    };
    mesh.compute_normals();
    mesh


}

#[allow(dead_code)]
fn smooth_triangle(phase:f32) -> f32{
    9.0/10.0 * (phase.cos() + (3.0*phase).cos()/9.0)
}

struct EversionMaterial {
    cull : Cull,
    id : u16,
    color : Srgba
}

impl Material for EversionMaterial{
    fn fragment_shader_source(&self, _lights: &[&dyn Light]) -> String {
        include_str!("surface.frag").to_string()
    }

    fn fragment_attributes(&self) -> FragmentAttributes {
        FragmentAttributes {
            position: true,
            normal : true,
            // uv : true,
            ..FragmentAttributes::NONE
        }
    }

    fn material_type(&self) -> MaterialType {
        MaterialType::Opaque
    }

    fn render_states(&self) -> RenderStates {
        RenderStates {
            
            write_mask: WriteMask::default(),
            blend : Blend::Disabled,
            cull: self.cull,
            ..Default::default()
        }
    }

    fn id(&self) -> u16 {
        self.id
    }

    fn use_uniforms(&self, program: &Program, camera: &Camera, _lights: &[&dyn Light]) {
        program.use_uniform("cameraPosition", camera.position());
        program.use_uniform("color", self.color.to_linear_srgb());
    }

    
}


#[allow(dead_code)]
pub fn main() {
    let window = Window::new(WindowSettings{
        max_size: Some((1280, 720)),
        ..Default::default()
    }).unwrap();

    // let mut assets = three_d_asset::io::load(&["assets/veranda_1k.hdr"]).unwrap();
        

    let ctx = window.gl();

    let mut cam = Camera::new_perspective(
        window.viewport(),
        vec3(0.0, 0.0, 3.5),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(
        Vector3::zero(),
         3.0, 
         5.0);

    
    // const LINT : f32 = 1.0;
    // let sun = DirectionalLight::new(&ctx, LINT, Srgba::RED, &Vector3::new(1.0, -1.0, 0.0).normalize());
    // let sun2 = DirectionalLight::new(&ctx, LINT, Srgba::GREEN, &Vector3::new(1.0, 1.0, 0.0).normalize());
    // let sun3 = DirectionalLight::new(&ctx, LINT, Srgba::BLUE, &Vector3::new(0.0,0.0, 1.0).normalize());
    // let amb = AmbientLight::new(&ctx, 0.1, Srgba::WHITE);
    // let skybox = Skybox::new_from_equirectangular(
    //     &ctx,
    //     &assets.deserialize("veranda_1k").unwrap(),
    // );
    let amb = AmbientLight::new(&ctx, 1.0, Srgba::WHITE);
    
    let mut gui = three_d::GUI::new(&ctx);


    let material_front = EversionMaterial{
        cull : Cull::Back,
        id : 0,
        color : Srgba::new(210, 205, 0, 255)
    };
    let material_back = EversionMaterial{
        cull : Cull::Front,
        id : 1,
        color : Srgba::new(30, 140, 232,255)
    };
    

    let mut omega : f32 = 3.0;
    // let mut time : f32 = 0.0;

    

    let mut eta : f32  = 1.0;
    // let mut xi : f32 = 3.0;

    // let mut alpha : f32 = 2.0;
    // let mut beta : f32 = 0.3;

    let mut n : usize = 2;

    // let mut lambda : f32 = 1.0;
    let mut lambdapower : f32 = 2.0;
    let mut alphapower : f32 = 2.0;
    let mut edgesize : f32 = 2.0;
    let mut betapower : f32 = 1.0;

    window.render_loop(move |mut frame_input|{
        let mut panel_width = 0.0;
        let time = (1.0+edgesize)*smooth_triangle((frame_input.accumulated_time * 0.0005) as f32);
        let t = time.clamp(-1.0,1.0) / Q;
        let lambda_raw = ((-time.abs() + 1.0 + edgesize)/edgesize).clamp(0.0,1.0);
            
        // let lambda = (lambda_raw*lambda_raw * (3.0-2.0*lambda_raw)).clamp(0.001,1.0);
        let lambda = lambda_raw.powf(lambdapower).clamp(0.001,1.0);
        let alpha = 2.0 * lambda.powf(alphapower).max(0.001);

        let xi = 3.0 * lambda;
        let lambda_bpow = lambda.powf(betapower);
        let beta = 0.3 *lambda_bpow + 1.0 *(1.0-lambda_bpow) ;

        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                use three_d::egui::*;
                SidePanel::left("side_panel").show(gui_context, |ui| {
                    use three_d::egui::*;
                    ui.heading("Debug Panel");
                    // ui.add(
                    //     Slider::new(&mut lambda, 0.0..=1.0).text("lambda")
                    // );
                    ui.add(
                        Slider::new(&mut n, 2..=6).text("n")
                    );
                    ui.add(
                        Slider::new(&mut omega, 0.0..=10.0).text("omega"),
                    );
                    // ui.add(
                    //     Slider::new(&mut time, -2.0/Q..=2.0/Q).text("Time"),
                    // );
                    ui.add(
                        Slider::new(&mut eta, 0.0..=3.0).text("eta"),
                    );
                    // ui.add(
                    //     Slider::new(&mut xi, 0.0..=10.0).text("xi"),
                    
                    // );
                    ui.add(
                        Slider::new(&mut lambdapower, 0.1..=5.0).text("lambdapower"),
                    );
                    ui.add(
                        Slider::new(&mut alphapower, 1.0..=3.0).text("alphapower"),
                    );
                    ui.add(
                        Slider::new(&mut edgesize, 0.0..=3.0).text("edgesize"),
                    );
                    ui.add(
                        Slider::new(&mut betapower, 0.0..=5.0).text("betapower"),
                    );
                    ui.label(format!("lambda {}, alpha {}, beta {}, xi {}",lambda,alpha,beta,xi));
                    
                    
                });
                panel_width = gui_context.used_rect().width();
            },
        );

        

        let geometry = 
            Mesh::new(&ctx,
                &make_surface(lambda,n,omega,t,eta,xi,alpha,beta));
            
        
        let lights : &[&dyn Light] = &[&amb];

        cam.set_viewport(frame_input.viewport);
        control.handle_events(&mut cam, &mut frame_input.events);
        frame_input.screen()
            .clear(ClearState::color_and_depth(0.5,0.5,0.5,1.0,1.0))
            .render_with_material(
                &material_back,&cam, &geometry, lights
            )
            .render_with_material(
                &material_front,&cam, &geometry, lights
            );
            // .write(|| gui.render()).unwrap();

        

        FrameOutput::default()
    });
}
